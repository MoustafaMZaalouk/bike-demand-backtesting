"""Training selection and locked evaluation, with exported audit trails."""

import argparse
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_pinball_loss

from bike_backtest.core import (
    CANDIDATES,
    DATA_SHA256,
    FEATURES,
    TEST,
    VALIDATION,
    Period,
    backtest,
    load_daily,
    metrics,
    paired_block_interval,
)


def train(daily, output, smoke=False):
    """Select only on validation. Smoke mode is a short execution check."""
    validation = Period("2012-01-01", "2012-01-07") if smoke else VALIDATION
    selection = []
    selected = {}
    for horizon in (1, 7):
        rows = []
        for name in CANDIDATES:
            predictions, _ = backtest(daily, validation, name, horizon=horizon)
            rows.append(
                {
                    "candidate": name,
                    "horizon_days": horizon,
                    **metrics(predictions.actual, predictions.tree),
                }
            )
        selected[str(horizon)] = min(
            rows, key=lambda row: (row["mae"], row["candidate"])
        )["candidate"]
        selection.extend(rows)
    pd.DataFrame(selection).to_csv(output / "validation_search.csv", index=False)
    config = {
        "selected": selected,
        "data_sha256": DATA_SHA256,
        "smoke": smoke,
        "selection_rule": "Minimum Jan-Mar 2012 validation MAE per horizon; candidate name breaks ties",
        "candidates": CANDIDATES,
    }
    (output / "selection.json").write_text(json.dumps(config, indent=4) + "\n")
    print(json.dumps(config, indent=4))


def evaluate(daily, output, smoke=False):
    config = json.loads((output / "selection.json").read_text())
    if config["smoke"] != smoke or config["data_sha256"] != DATA_SHA256:
        raise ValueError("Selection mode/data mismatch; rerun train.")
    period = Period("2012-04-01", "2012-04-07") if smoke else TEST
    records, audits, monthly, summary_rows = [], [], [], []
    for horizon in (1, 7):
        name = config["selected"][str(horizon)]
        frame, audit = backtest(daily, period, name, horizon=horizon)
        poisson, _ = backtest(daily, period, "poisson", horizon=horizon)
        lower, _ = backtest(daily, period, name, horizon=horizon, quantile=0.1)
        upper, _ = backtest(daily, period, name, horizon=horizon, quantile=0.9)
        frame["poisson"] = poisson.tree
        # Rearrangement prevents quantile crossing, without using held-out labels.
        frame["lower_80"] = np.minimum(lower.tree, upper.tree)
        frame["upper_80"] = np.maximum(lower.tree, upper.tree)
        records.append(frame)
        audit["horizon_days"] = horizon
        audits.append(audit)
        by_model = {}
        for model in ("persistence", "weekly_baseline", "poisson", "tree"):
            by_model[model] = metrics(frame.actual, frame[model])
        monthly_groups = frame.groupby(frame.date.dt.to_period("M"))
        for month, rows in monthly_groups:
            for model in by_model:
                monthly.append(
                    {
                        "month": str(month),
                        "horizon_days": horizon,
                        "model": model,
                        "days": len(rows),
                        **metrics(rows.actual, rows[model]),
                    }
                )
        covered = (frame.actual >= frame.lower_80) & (frame.actual <= frame.upper_80)
        summary_rows.append(
            {
                "horizon_days": horizon,
                "selected_model": name,
                "days": len(frame),
                "metrics": by_model,
                "mae_reduction_vs_weekly_percent": 100
                * (1 - by_model["tree"]["mae"] / by_model["weekly_baseline"]["mae"]),
                "interval": {
                    "nominal_coverage": 0.8,
                    "observed_coverage": float(covered.mean()),
                    "mean_width": float((frame.upper_80 - frame.lower_80).mean()),
                    "pinball_0.1": mean_pinball_loss(
                        frame.actual, frame.lower_80, alpha=0.1
                    ),
                    "pinball_0.9": mean_pinball_loss(
                        frame.actual, frame.upper_80, alpha=0.9
                    ),
                },
                "paired_block_bootstrap": paired_block_interval(frame),
            }
        )
    pd.concat(records).to_csv(
        output / "test_predictions.csv", index=False, float_format="%.10f"
    )
    pd.concat(audits).to_csv(output / "refit_audit.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_metrics.csv", index=False)
    summary = {
        "data_sha256": DATA_SHA256,
        "raw_rows": len(daily),
        "features": FEATURES,
        "validation_start": VALIDATION.start,
        "validation_end": VALIDATION.end,
        "test_start": period.start,
        "test_end": period.end,
        "smoke": smoke,
        "horizons": summary_rows,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=4) + "\n")
    print(json.dumps(summary, indent=4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "eval", "all"])
    parser.add_argument("--data", type=Path, default=Path("data/day.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    daily = load_daily(args.data)
    if args.smoke:
        daily = daily.loc[:"2012-04-07"].copy()
    if args.stage in ("train", "all"):
        train(daily, args.output, args.smoke)
    if args.stage in ("eval", "all"):
        evaluate(daily, args.output, args.smoke)


if __name__ == "__main__":
    main()
