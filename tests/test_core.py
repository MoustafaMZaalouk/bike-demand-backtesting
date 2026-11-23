from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_backtest.core import (
    FEATURES,
    Period,
    backtest,
    load_daily,
    make_features,
    metrics,
    paired_block_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic(days=140):
    dates = pd.date_range("2011-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {"cnt": np.arange(days, dtype=float), "holiday": 0, "workingday": 1},
        index=dates,
    )


class MeanSpy:
    def __init__(self):
        self.fit_dates = None
        self.fit_labels = None
        self.predict_dates = None

    def fit(self, x, y):
        self.fit_dates = x.index.copy()
        self.fit_labels = y.copy()
        self.mean = y.mean()
        return self

    def predict(self, x):
        self.predict_dates = x.index.copy()
        return np.full(len(x), self.mean)


def test_hand_calculated_lags_and_shifted_rolling_windows():
    features = make_features(synthetic())
    row = features.iloc[28]
    assert row.lag_1 == 27
    assert row.lag_7 == 21
    assert row.lag_14 == 14
    assert row.mean_7 == np.mean(np.arange(21, 28))
    assert row.mean_28 == np.mean(np.arange(28))
    assert row.std_7 == pytest.approx(np.std(np.arange(21, 28)))
    assert features.mean_28.iloc[:28].isna().all()


def test_calendar_encoding_matches_python_calendar():
    daily = synthetic()
    date = pd.Timestamp("2011-01-31")  # Monday
    features = make_features(daily)
    assert date.dayofweek == 0
    assert features.loc[date, "weekday_sin"] == pytest.approx(0)
    assert features.loc[date, "weekday_cos"] == pytest.approx(1)


def test_changing_current_and_future_targets_cannot_change_current_features():
    daily = synthetic()
    target = daily.index[60]
    original = make_features(daily)
    changed = daily.copy()
    changed.loc[target:, "cnt"] = 999999
    pd.testing.assert_frame_equal(
        original.loc[:target], make_features(changed).loc[:target]
    )
    assert (
        original.loc[daily.index[61], "lag_1"]
        != make_features(changed).loc[daily.index[61], "lag_1"]
    )


def test_weather_and_target_components_are_never_used():
    daily = load_daily(ROOT / "data/day.csv")
    original = make_features(daily)
    for column in [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "weathersit",
        "casual",
        "registered",
        "instant",
    ]:
        daily[column] = -999999
    pd.testing.assert_frame_equal(original, make_features(daily))
    assert list(original.columns) == FEATURES
    assert "cnt" not in FEATURES


def test_monthly_fit_excludes_current_month_and_baseline_is_exact():
    daily = synthetic()
    fitted = []

    def factory():
        model = MeanSpy()
        fitted.append(model)
        return model

    predictions, audit = backtest(
        daily, Period("2011-03-01", "2011-04-10"), "unused", factory
    )
    assert len(fitted) == 2
    assert fitted[0].fit_dates.max() == pd.Timestamp("2011-02-28")
    assert fitted[1].fit_dates.max() == pd.Timestamp("2011-03-31")
    for model in fitted:
        assert model.fit_dates.max() < model.predict_dates.min()
        pd.testing.assert_series_equal(
            model.fit_labels, daily.loc[model.fit_dates, "cnt"]
        )
        expected = model.fit_labels.mean()
        assert np.all(
            predictions.loc[predictions.date.isin(model.predict_dates), "tree"]
            == expected
        )
    expected_baseline = daily.cnt.shift(7).loc[predictions.date].to_numpy()
    np.testing.assert_array_equal(predictions.weekly_baseline, expected_baseline)
    assert (audit.train_end < audit.predict_start).all()
    assert (predictions.feature_history_end < predictions.date).all()


def test_real_tree_predictions_do_not_look_forward():
    daily = load_daily(ROOT / "data/day.csv")
    period = Period("2012-04-01", "2012-04-10")
    original, _ = backtest(daily, period, "small_tree")
    changed = daily.copy()
    changed.loc["2012-04-05":, "cnt"] = 123456
    altered, _ = backtest(changed, period, "small_tree")
    np.testing.assert_array_equal(original.tree.iloc[:5], altered.tree.iloc[:5])
    np.testing.assert_array_equal(
        original.weekly_baseline.iloc[:5], altered.weekly_baseline.iloc[:5]
    )


def test_metrics_are_hand_calculated_not_only_shapes():
    result = metrics([0, 10, 20], [2, 8, 25])
    assert result["mae"] == 3
    assert result["rmse"] == pytest.approx(np.sqrt(11))
    assert result["wape_percent"] == 30
    assert np.isnan(metrics([0], [0])["wape_percent"])
    with pytest.raises(ValueError):
        metrics([1], [1, 2])


def test_bootstrap_preserves_constant_paired_gain():
    frame = pd.DataFrame(
        {
            "actual": np.full(28, 10),
            "weekly_baseline": np.full(28, 6),
            "tree": np.full(28, 9),
        }
    )
    result = paired_block_interval(frame)
    assert result["mean_mae_gain"] == result["lower_95"] == result["upper_95"] == 3
    assert result == paired_block_interval(frame)


def test_real_data_integrity():
    daily = load_daily(ROOT / "data/day.csv")
    assert len(daily) == 731
    assert daily.index.min() == pd.Timestamp("2011-01-01")
    assert daily.index.max() == pd.Timestamp("2012-12-31")
    assert (daily.cnt == daily.casual + daily.registered).all()
    assert daily.cnt.iloc[0] == 985


def test_rejects_data_tampering_and_calendar_gaps(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("not the dataset")
    with pytest.raises(ValueError, match="checksum"):
        load_daily(path)
    daily = synthetic().drop(pd.Timestamp("2011-01-15"))
    daily.rename_axis("dteday").to_csv(path)
    with pytest.raises(ValueError, match="Missing calendar"):
        load_daily(path, verify_hash=False)


def test_seven_day_horizon_has_seven_day_information_gap():
    daily = synthetic()
    features = make_features(daily, horizon=7)
    row = features.iloc[40]
    assert row.lag_1 == 33
    assert row.lag_7 == 27
    assert row.mean_7 == np.mean(np.arange(27, 34))
    fitted = []

    def factory():
        model = MeanSpy()
        fitted.append(model)
        return model

    predictions, audit = backtest(
        daily, Period("2011-04-01", "2011-04-10"), "unused", factory, horizon=7
    )
    assert fitted[0].fit_dates.max() == pd.Timestamp("2011-03-25")
    assert set(fitted[0].fit_dates).isdisjoint(predictions.date)
    assert (audit.train_end <= audit.predict_start - pd.Timedelta(days=7)).all()
    target = pd.Timestamp("2011-04-01")
    changed = daily.copy()
    changed.loc["2011-03-26":, "cnt"] = 500000
    pd.testing.assert_series_equal(
        features.loc[target], make_features(changed, 7).loc[target]
    )


def test_binary_flags_and_fractional_counts_are_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    daily = synthetic()
    daily.iloc[0, daily.columns.get_loc("holiday")] = 3
    daily.rename_axis("dteday").to_csv(path)
    with pytest.raises(ValueError, match="binary"):
        load_daily(path, verify_hash=False)
    daily = synthetic()
    daily.iloc[0, daily.columns.get_loc("cnt")] = 0.5
    daily.rename_axis("dteday").to_csv(path)
    with pytest.raises(ValueError, match="integers"):
        load_daily(path, verify_hash=False)


def test_interval_bounds_and_committed_metrics_match_predictions():
    import json

    predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
    summary = json.loads((ROOT / "results/metrics.json").read_text())
    assert (predictions.lower_80 <= predictions.upper_80).all()
    for horizon in summary["horizons"]:
        rows = predictions.loc[predictions.horizon_days == horizon["horizon_days"]]
        assert len(rows) == 275
        assert rows.date.is_unique
        for model, expected in horizon["metrics"].items():
            measured = metrics(rows.actual, rows[model])
            for key, value in measured.items():
                assert value == pytest.approx(expected[key], abs=1e-6)
        coverage = (
            (rows.actual >= rows.lower_80) & (rows.actual <= rows.upper_80)
        ).mean()
        assert coverage == pytest.approx(horizon["interval"]["observed_coverage"])


def test_missing_schema_and_duplicate_dates_are_rejected(tmp_path):
    path = tmp_path / "bad-schema.csv"
    daily = synthetic().drop(columns=["workingday"])
    daily.rename_axis("dteday").to_csv(path)
    with pytest.raises(ValueError, match="Missing required"):
        load_daily(path, verify_hash=False)
    daily = synthetic()
    daily = pd.concat([daily.iloc[:1], daily])
    daily.rename_axis("dteday").to_csv(path)
    with pytest.raises(ValueError, match="sorted and unique"):
        load_daily(path, verify_hash=False)
