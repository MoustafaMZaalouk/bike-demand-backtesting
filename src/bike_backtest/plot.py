"""Rebuild the overview from exported predictions; no retraining required."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    predictions = pd.read_csv(
        args.results / "test_predictions.csv", parse_dates=["date"]
    )
    predictions = predictions.loc[predictions.horizon_days == 1].copy()
    monthly = pd.read_csv(args.results / "monthly_metrics.csv")
    monthly = monthly.loc[
        (monthly.horizon_days == 1) & monthly.model.isin(["weekly_baseline", "tree"])
    ].copy()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.4, 1]}
    )
    figure.set_facecolor("#f7f8fa")
    for axis in axes:
        axis.set_facecolor("#f7f8fa")
        axis.grid(axis="y", alpha=0.2)
    for column, label, color, width in [
        ("actual", "Observed", "#263348", 1.4),
        ("weekly_baseline", "Same day last week", "#a1a8b4", 0.9),
        ("tree", "Gradient-boosted tree", "#057b83", 1.2),
    ]:
        axes[0].plot(
            predictions.date,
            predictions[column],
            label=label,
            color=color,
            linewidth=width,
        )
    axes[0].fill_between(
        predictions.date,
        predictions.lower_80,
        predictions.upper_80,
        color="#057b83",
        alpha=0.12,
        label="Nominal 80% interval",
    )
    axes[0].set_title(
        "Daily forecasts on the held-out test period", loc="left", fontsize=13
    )
    axes[0].set_ylabel("Rentals / day")
    axes[0].legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    pivot = monthly.pivot(index="month", columns="model", values="mae")
    positions = list(range(len(pivot)))
    axes[1].bar(
        [p - 0.18 for p in positions],
        pivot.weekly_baseline,
        width=0.36,
        label="Weekly baseline",
        color="#a1a8b4",
    )
    axes[1].bar(
        [p + 0.18 for p in positions],
        pivot.tree,
        width=0.36,
        label="Tree",
        color="#057b83",
    )
    axes[1].set_xticks(positions, [month[5:] for month in pivot.index])
    axes[1].set_xlabel("Target month in 2012")
    axes[1].set_ylabel("MAE (rentals / day)")
    axes[1].set_title("Monthly error: lower is better", loc="left", fontsize=13)
    axes[1].legend(frameon=False, ncol=2)
    figure.suptitle(
        "BIKE DEMAND / NEXT-DAY BACKTEST",
        x=0.075,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    figure.text(
        0.075,
        0.025,
        "UCI Bike Sharing, 2011-2012 | Monthly refits, daily updated lags | No target-day weather",
        fontsize=9,
        color="#4c5566",
    )
    figure.tight_layout(rect=(0.025, 0.055, 1, 0.95), h_pad=2)
    figure.savefig(
        args.results / "backtest.png", dpi=160, facecolor=figure.get_facecolor()
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
