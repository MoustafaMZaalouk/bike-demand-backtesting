"""Leakage-aware daily features and expanding-window backtesting."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_SHA256 = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be"
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FEATURES = [
    "weekday_sin",
    "weekday_cos",
    "annual_sin",
    "annual_cos",
    "trend_days",
    "holiday",
    "workingday",
    "lag_1",
    "lag_7",
    "lag_14",
    "mean_7",
    "mean_28",
    "std_7",
]
CANDIDATES = {
    "small_tree": {"num_leaves": 7, "reg_lambda": 10.0},
    "medium_tree": {"num_leaves": 15, "reg_lambda": 10.0},
    "regularized_tree": {"num_leaves": 7, "reg_lambda": 100.0},
}


@dataclass(frozen=True)
class Period:
    """Inclusive evaluation dates; all earlier usable rows are training rows."""

    start: str
    end: str


VALIDATION = Period("2012-01-01", "2012-03-31")
TEST = Period("2012-04-01", "2012-12-31")


def load_daily(path: Path, verify_hash: bool = True) -> pd.DataFrame:
    """Validate the exact archived daily file and its calendar integrity."""
    if verify_hash and hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum mismatch; refuse a silent data change.")
    daily = pd.read_csv(path, parse_dates=["dteday"]).set_index("dteday")
    required = {"cnt", "holiday", "workingday"}
    if not required.issubset(daily.columns):
        raise ValueError("Missing required columns.")
    if not daily.index.is_monotonic_increasing or not daily.index.is_unique:
        raise ValueError("Dates must be sorted and unique.")
    expected = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    if not daily.index.equals(expected):
        raise ValueError("Missing calendar days would make row shifts unsafe.")
    if not pd.api.types.is_numeric_dtype(daily["cnt"]):
        raise ValueError("Counts must be numeric.")
    if daily[list(required)].isna().any().any() or (daily["cnt"] < 0).any():
        raise ValueError("Missing required values or negative rental counts.")
    if (
        not daily["holiday"].isin([0, 1]).all()
        or not daily["workingday"].isin([0, 1]).all()
    ):
        raise ValueError("Calendar flags must be binary.")
    if not np.isfinite(daily["cnt"]).all() or not (daily["cnt"] % 1 == 0).all():
        raise ValueError("Counts must be finite integers.")
    return daily


def make_features(daily: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """For target D, demand inputs stop at D-horizon; calendar is known ahead.

    Lag suffixes are offsets from the one-day setup: lag_1 means D-horizon,
    lag_7 means D-(horizon+6), and lag_14 means D-(horizon+13).
    """
    if horizon not in (1, 7):
        raise ValueError("Supported daily horizons are 1 and 7 days.")
    dates = daily.index
    features = pd.DataFrame(index=dates)
    weekday = dates.dayofweek
    annual_phase = (dates.dayofyear - 1) / 365.25
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["annual_sin"] = np.sin(2 * np.pi * annual_phase)
    features["annual_cos"] = np.cos(2 * np.pi * annual_phase)
    features["trend_days"] = (dates - pd.Timestamp("2011-01-01")).days
    features["holiday"] = daily["holiday"]
    features["workingday"] = daily["workingday"]
    for lag in (1, 7, 14):
        features[f"lag_{lag}"] = daily["cnt"].shift(lag + horizon - 1)
    history = daily["cnt"].shift(horizon)
    features["mean_7"] = history.rolling(7, min_periods=7).mean()
    features["mean_28"] = history.rolling(28, min_periods=28).mean()
    features["std_7"] = history.rolling(7, min_periods=7).std(ddof=0)
    return features[FEATURES]


def make_model(name: str, quantile: float | None = None):
    """LightGBM trained from scratch, or a scaled Poisson regression."""
    if name == "poisson":
        return make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000, tol=1e-6)
        )
    parameters = dict(
        objective="regression",
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=15,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **CANDIDATES[name],
    )
    if quantile is not None:
        parameters.update(objective="quantile", alpha=quantile)
    return LGBMRegressor(**parameters)


def backtest(
    daily: pd.DataFrame,
    period: Period,
    model_name: str,
    model_factory=None,
    horizon: int = 1,
    quantile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly expanding refits with fixed horizon and chronological labels.

    For target D, the information cutoff is D-horizon. The first target of
    each month determines the refit's training cutoff. Subsequent forecasts
    use newly available lag inputs, not new model fits. This is an online
    fixed-horizon backtest, not a batch multi-output month forecast.
    """
    x = make_features(daily, horizon)
    eligible = x.notna().all(axis=1)
    selected = x.index[(x.index >= period.start) & (x.index <= period.end)]
    if len(selected) == 0:
        raise ValueError("No evaluation dates.")
    if not eligible.loc[selected].all():
        raise ValueError("Evaluation includes insufficient feature history.")
    factory = model_factory or (lambda: make_model(model_name, quantile))
    predictions = []
    audit = []
    for month in selected.to_period("M").unique():
        targets = selected[selected.to_period("M") == month]
        cutoff = targets.min()
        training = x.index[eligible & (x.index <= cutoff - pd.Timedelta(days=horizon))]
        if len(training) < 28:
            raise ValueError("Insufficient training history.")
        model = factory()
        model.fit(x.loc[training], daily.loc[training, "cnt"])
        prediction = np.maximum(0, model.predict(x.loc[targets]))
        predictions.append(
            pd.DataFrame(
                {
                    "date": targets,
                    "actual": daily.loc[targets, "cnt"].to_numpy(),
                    "weekly_baseline": daily["cnt"].shift(7).loc[targets].to_numpy(),
                    "persistence": daily["cnt"].shift(horizon).loc[targets].to_numpy(),
                    "horizon_days": horizon,
                    "tree": prediction,
                    "train_end": training.max(),
                    "feature_history_end": targets - pd.Timedelta(days=horizon),
                }
            )
        )
        audit.append(
            {
                "refit_date": cutoff,
                "train_start": training.min(),
                "train_end": training.max(),
                "train_rows": len(training),
                "predict_start": targets.min(),
                "predict_end": targets.max(),
                "predict_rows": len(targets),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit)


def metrics(actual, predicted) -> dict[str, float]:
    """MAE/RMSE in rentals per day; WAPE in percent, not fraction."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted must be equal, nonempty shapes.")
    error = actual - predicted
    denominator = np.abs(actual).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_percent": float(100 * np.abs(error).sum() / denominator)
        if denominator
        else float("nan"),
    }


def paired_block_interval(predictions: pd.DataFrame) -> dict:
    """Descriptive paired MAE gain interval, resampling 7-day moving blocks.

    Positive gains favor the tree. Conditional on this fitted experiment, not
    an interval for future deployment or repeated model selection.
    """
    gain = (
        np.abs(predictions.actual - predictions.weekly_baseline)
        - np.abs(predictions.actual - predictions.tree)
    ).to_numpy()
    if len(gain) < 7:
        raise ValueError("Need at least one seven-day block.")
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(gain) - 6, size=(2000, int(np.ceil(len(gain) / 7))))
    indices = (starts[..., None] + np.arange(7)).reshape(2000, -1)[:, : len(gain)]
    samples = gain[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_mae_gain": float(gain.mean()),
        "lower_95": float(low),
        "upper_95": float(high),
        "block_days": 7,
        "resamples": 2000,
    }
