"""Required poisoned-future and selector contract tests (no ADV score reads)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from tfb.lane_q.windows import (Release, candidate_windows, grid_from_frame,
                               horizon_labels, interpolated_labels, online_view,
                               frozen_windows, Network)


def features(view):
    observations = view.observations()
    valid = observations[observations.speed_kmh.notna()]
    last = valid.sort_values(["timestamp", "link_id"]).groupby("link_id").tail(1)
    return {"history": view.history, "mainline": view.mainline, "ramps": view.ramps,
            "observations": observations, "last": last.reset_index(drop=True),
            "condition": view.condition(),
            "previous": list(view.previous_days()),
            "training": list(view.training_reference())}


def assert_features_equal(test, left, right):
    test.assertEqual(left.keys(), right.keys())
    for key in left:
        if isinstance(left[key], pd.DataFrame):
            pd.testing.assert_frame_equal(left[key], right[key], check_exact=True)
        elif isinstance(left[key], list):
            test.assertEqual([d for d, _ in left[key]], [d for d, _ in right[key]])
            for (_, a), (_, b) in zip(left[key], right[key]):
                pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            test.assertEqual(left[key], right[key])


class OnlineTests(unittest.TestCase):
    def fixture(self, root, official_train=False):
        panel = "TEST_N"
        network = root / "corridors" / panel / "network"
        network.mkdir(parents=True)
        pd.DataFrame({"link_id": ["A", "B"], "order_index": [0, 1]}).to_csv(network / "lwr_mainline_topology.csv", index=False)
        pd.DataFrame({"link_id": ["A", "B"], "free_speed_kmh": [100., 100.]}).to_csv(network / "fd_parameters.csv", index=False)
        for split in ["train", "validation", "private"]:
            for date in ["2031-01-01", "2031-01-02", "2031-01-03"]:
                times = pd.date_range(date, periods=288, freq="5min", tz="UTC")
                frame = pd.DataFrame({"timestamp": np.repeat(times.strftime("%Y-%m-%dT%H:%M:%SZ"), 2),
                                      "link_id": np.tile(["A", "B"], 288),
                                      "speed_kmh": 90., "flow_vph": 1000., "occupancy": .1,
                                      "pct_observed": 100, "is_score_eligible": 1})
                frame.loc[(frame.timestamp >= date + "T11:30:00Z") & frame.link_id.eq("A"), "speed_kmh"] = 40.
                for source in ["mainline_states_masked", "ramp_states"] + (["mainline_states"] if split == "train" else []):
                    dest = root / "corridors" / panel / split / source / ("mask_regime=R1" if source == "mainline_states_masked" else "days")
                    dest.mkdir(parents=True, exist_ok=True)
                    values = frame.copy()
                    if source == "ramp_states":
                        values = values.drop(columns=["speed_kmh", "occupancy"]).rename(columns={"link_id": "ramp_link_id"})
                    elif source == "mainline_states_masked":
                        values["mask_regime"] = "R1"
                        values.loc[values.link_id.eq("B"), "speed_kmh"] = np.nan
                    values.to_parquet(dest / ("synthetic_mainline_" + date.replace("-", "_") + ".parquet"), index=False)
            if split != "train" or official_train:
                dest = root / "task2" / panel / split
                dest.mkdir(parents=True)
                pd.DataFrame({"window_id": ["join-only"], "date": ["2031-01-02"],
                              "forecast_origin": ["2031-01-02T12:00:00Z"]}).to_csv(dest / "window_index.csv", index=False)
                history = frame.copy()
                history.timestamp = history.timestamp.str.replace("2031-01-03", "2031-01-02")
                history = history[history.timestamp.between("2031-01-02T11:00:00Z", "2031-01-02T11:55:00Z")]
                # Hostile extra rows with the same window key must be excluded.
                future = history.iloc[:2].copy()
                future.timestamp = "2031-01-02T13:00:00Z"
                history = pd.concat([history, future], ignore_index=True)
                history["window_id"] = "join-only"
                history.to_parquet(dest / "window_history.parquet", index=False)
        return panel

    def poison(self, root, split, value, origin="2031-01-02T12:00:00Z"):
        origin = pd.Timestamp(origin)
        for path in (root / "corridors" / "TEST_N" / split).rglob("*.parquet"):
            frame = pd.read_parquet(path)
            future = pd.to_datetime(frame.timestamp, utc=True).gt(origin)
            for col in ["speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]:
                if col in frame:
                    frame.loc[future, col] = value
            frame.to_parquet(path, index=False)
        history_path = root / "task2" / "TEST_N" / split / "window_history.parquet"
        if history_path.exists():
            frame = pd.read_parquet(history_path)
            future = pd.to_datetime(frame.timestamp, utc=True).gt(origin)
            for col in ["speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]:
                frame.loc[future, col] = value
            frame.to_parquet(history_path, index=False)

    def test_poisoned_future_generated_and_official_all_splits(self):
        for official in [False, True]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                panel = self.fixture(root, official)
                for split in ["train", "validation", "private"]:
                    with self.subTest(official_train=official, split=split):
                        args = (panel, split, "2031-01-02", "2031-01-02T12:00:00Z")
                        original = features(online_view(*args, release_root=root))
                        for value in [-9e8, np.nan]:
                            self.poison(root, split, value)
                            poisoned = features(online_view(*args, release_root=root))
                            assert_features_equal(self, original, poisoned)
                        self.assertLessEqual(original["observations"].timestamp.max(), pd.Timestamp(args[3]))
                        self.assertNotIn("window_id", original["history"].columns)
                        self.assertEqual([d for d, _ in original["previous"]], ["2031-01-01"])

    def test_official_history_and_masked_T_are_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            panel = self.fixture(root, True)
            view = online_view(panel, "train", "2031-01-02", "2031-01-02T12:00:00Z", release_root=root)
            self.assertEqual(view.history.timestamp.nunique(), 12)
            self.assertEqual(view.history.timestamp.max(), pd.Timestamp("2031-01-02T11:55:00Z"))
            at_T = view.observations().query("timestamp == @view.origin")
            self.assertTrue(at_T.loc[at_T.link_id.eq("B"), "speed_kmh"].isna().all())
            self.assertEqual(view.condition(), "queue_ongoing")
            with self.assertRaises(ValueError):
                online_view(panel, "validation", "2031-01-02", "2031-01-02T12:05:00Z", release_root=root).history
            with self.assertRaises(ValueError):
                online_view(panel, "train", "2031-01-03", "2031-01-02T12:00:00Z", release_root=root)


class SelectorTests(unittest.TestCase):
    def test_threshold_null_eligibility_and_condition(self):
        network = Network("TEST_N", ("A",), np.array([100.]), pd.DataFrame())
        frame = pd.DataFrame({"timestamp": pd.date_range("2031-01-01", periods=288, freq="5min", tz="UTC"),
                              "link_id": "A", "speed_kmh": 90., "is_score_eligible": 1})
        frame.loc[100:110, "speed_kmh"] = 60.
        frame.loc[99, "speed_kmh"] = np.nan
        day = grid_from_frame(frame, network, "2031-01-01")
        candidates = candidate_windows(day)
        onset = candidates[candidates.condition.eq("queue_onset")]
        self.assertEqual(onset.forecast_origin.tolist(), ["2031-01-01T07:50:00Z"])
        labels = horizon_labels(day, onset.iloc[0].forecast_origin)
        self.assertEqual(labels.queue_true.sum(), 1)
        self.assertEqual(labels.is_score_eligible.sum(), 5)
        filled = interpolated_labels(day, onset.iloc[0].forecast_origin, network)
        self.assertEqual(filled.is_score_eligible.sum(), 6)
        self.assertEqual(filled.label_observed.sum(), 5)
        frame.loc[100:110, "is_score_eligible"] = 0
        eligible_day = grid_from_frame(frame, network, "2031-01-01")
        self.assertTrue(candidate_windows(eligible_day, eligible_only=True).empty)
        self.assertFalse(candidate_windows(eligible_day).empty)

    def test_frozen_lists_reproducible_without_row_order(self):
        rows = []
        for d in range(1, 13):
            for hour, cond in [(8, "queue_onset"), (9, "queue_ongoing"), (16, "queue_ongoing")]:
                t = f"2031-01-{d:02d}T{hour:02d}:00:00Z"
                rows.append({"panel": "TEST_N", "split": "train", "date": t[:10], "forecast_origin": t, "condition": cond})
        frame = pd.DataFrame(rows)
        a = frozen_windows(frame)
        b = frozen_windows(frame.sample(frac=1, random_state=18))
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        self.assertEqual(a.condition.value_counts().to_dict(), {"queue_onset": 5, "queue_ongoing": 5})
        for _, group in a.groupby("condition"):
            self.assertTrue(pd.to_datetime(group.forecast_origin, utc=True).sort_values().diff().dropna().ge(pd.Timedelta(minutes=360)).all())


    def test_spacing_emulation_allows_two_onsets_same_day(self):
        rows = []
        for d in range(1, 13):
            for hour, cond in [(5, "queue_onset"), (12, "queue_onset"), (9, "queue_ongoing"), (16, "queue_ongoing")]:
                t = f"2031-01-{d:02d}T{hour:02d}:00:00Z"
                rows.append({"panel": "TEST_N", "split": "train", "date": t[:10], "forecast_origin": t,
                             "condition": cond, "history_last_step_persistence_proxy_iou": .5})
        frame = pd.DataFrame(rows)
        first_daily = frozen_windows(frame)
        spaced = frozen_windows(frame, first_onset_per_day=False)
        expected = ["2031-01-01T05:00:00Z", "2031-01-01T12:00:00Z", "2031-01-02T05:00:00Z",
                    "2031-01-02T12:00:00Z", "2031-01-03T05:00:00Z"]
        self.assertEqual(spaced[spaced.condition.eq("queue_onset")].forecast_origin.tolist(), expected)
        self.assertEqual(first_daily[first_daily.condition.eq("queue_onset")].date.nunique(), 5)
        self.assertTrue(spaced[spaced.condition.eq("queue_ongoing")].history_last_step_persistence_proxy_iou.le(.9).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
