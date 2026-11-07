"""Required prediction-level future poisoning and keyed output contracts."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from . import test_windows
from .predict import KEYS, fit_panel_prior, predict_queue, predict_window
from .windows import Release, iso, timestamp


class PredictionTests(unittest.TestCase):
    def fixture(self, root):
        helper = test_windows.OnlineTests()
        panel = helper.fixture(root, False)
        raw = Release(root).read_day(panel, "train", "2031-01-02", "mainline_states")
        windows = [{"window_id": "onset_join", "date": "2031-01-02", "forecast_origin": "2031-01-02T11:20:00Z"},
                   {"window_id": "ongoing_join", "date": "2031-01-02", "forecast_origin": "2031-01-02T12:00:00Z"}]
        for split in ["train", "validation", "private"]:
            dest = root / "task2" / panel / split
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(windows).to_csv(dest / "window_index.csv", index=False)
            histories, templates = [], []
            for w in windows:
                t = timestamp(w["forecast_origin"])
                history = raw[raw.timestamp.between(t-pd.Timedelta(minutes=60), t-pd.Timedelta(minutes=5))].copy()
                extra = history.iloc[:2].copy()
                extra.timestamp = t+pd.Timedelta(minutes=60)
                history = pd.concat([history, extra], ignore_index=True)
                history.timestamp = [iso(t) for t in history.timestamp]
                history["window_id"] = w["window_id"]
                histories.append(history)
                templates.append(pd.DataFrame({"window_id": w["window_id"],
                                                "timestamp": np.repeat([iso(t+pd.Timedelta(minutes=5*i)) for i in range(1,7)], 2),
                                                "link_id": np.tile(["A", "B"], 6)}))
            pd.concat(histories, ignore_index=True).to_parquet(dest / "window_history.parquet", index=False)
            pd.concat(templates, ignore_index=True).to_csv(dest / "sample_submission_queue.csv", index=False)
        model = fit_panel_prior([{"hour": 11, "day_type": "weekday", "target_links": ["A"]}], ["A", "B"])
        cfg = {"release_root": str(root), "prior": {"model_version": 1, "panels": {panel: model}}}
        return helper, panel, cfg

    def test_predictor_poison_all_splits_and_conditions(self):
        for split in ["train", "validation", "private"]:
            for window_id, origin in [("onset_join", "2031-01-02T11:20:00Z"), ("ongoing_join", "2031-01-02T12:00:00Z")]:
                with self.subTest(split=split, window=window_id), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    helper, panel, cfg = self.fixture(root)
                    dest = root / "task2" / panel / split
                    index = pd.read_csv(dest / "window_index.csv")
                    index[index.window_id.eq(window_id)].to_csv(dest / "window_index.csv", index=False)
                    template = pd.read_csv(dest / "sample_submission_queue.csv")
                    template[template.window_id.eq(window_id)].to_csv(dest / "sample_submission_queue.csv", index=False)
                    before = predict_queue(panel, split, cfg)
                    expected = np.zeros(12, dtype=np.int8)
                    if window_id == "onset_join":
                        expected[10] = 1
                    else:
                        expected[::2] = 1
                    np.testing.assert_array_equal(before.queue_pred, expected)
                    for value in [-9e8, np.nan]:
                        helper.poison(root, split, value, origin)
                        after = predict_queue(panel, split, cfg)
                        pd.testing.assert_frame_equal(before, after, check_exact=True)
                    self.assertEqual(before.queue_pred.dtype, np.dtype(np.int8))

    def test_window_ids_and_row_order_are_only_joins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, panel, cfg = self.fixture(root)
            original = predict_queue(panel, "private", cfg).sort_values(KEYS).reset_index(drop=True)
            dest = root / "task2" / panel / "private"
            mapping = {"onset_join": "arbitrary_label_z", "ongoing_join": "arbitrary_label_a"}
            for name in ["window_index.csv", "sample_submission_queue.csv", "window_history.parquet"]:
                path = dest / name
                frame = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
                frame.window_id = frame.window_id.map(mapping)
                frame = frame.sample(frac=1, random_state=53).reset_index(drop=True)
                if name.endswith(".parquet"):
                    frame.to_parquet(path, index=False)
                else:
                    frame.to_csv(path, index=False)
            after = predict_queue(panel, "private", cfg)
            after.window_id = after.window_id.map({v: k for k, v in mapping.items()})
            after = after.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(original, after, check_exact=True)

    def test_masked_T_overrides_earlier_queued_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, panel, cfg = self.fixture(root)
            release = Release(root)
            path = release.files(panel, "train", "mainline_states_masked")["2031-01-02"]
            frame = pd.read_parquet(path)
            frame.loc[frame.timestamp.eq("2031-01-02T12:00:00Z") & frame.link_id.eq("A"), "speed_kmh"] = 90.
            frame.to_parquet(path, index=False)
            prediction = predict_window(panel, "train", "2031-01-02", "2031-01-02T12:00:00Z", "opaque", cfg)
            self.assertTrue(prediction.queue_pred.eq(0).all())

    def test_prior_fit_is_invariant_to_training_row_order(self):
        links = ["A", "B", "C", "D", "E"]
        records = [{"hour": 11, "day_type": "weekday", "target_links": cells}
                   for cells in [["A", "B"], ["B", "C"], ["E"], ["E"]]]
        a = fit_panel_prior(records, links)
        b = fit_panel_prior(list(reversed(records)), links)
        self.assertEqual(a, b)
        self.assertEqual([s["positions"] for s in a["sites"]], [[0,1,2], [4]])
        self.assertEqual(a["contexts"]["all"]["target_links"], ["A", "B"])


    def test_evaluation_cache_future_poison(self):
        from .evaluation_cache import EvaluationRelease
        for window_id, origin in [("onset_join", "2031-01-02T11:20:00Z"), ("ongoing_join", "2031-01-02T12:00:00Z")]:
            with self.subTest(window=window_id), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                helper, panel, cfg = self.fixture(root)
                release = EvaluationRelease(root)
                # Preload full-day evaluation labels. They must not become features.
                release.train_grid(panel, "2031-01-02")
                args = (panel, "train", "2031-01-02", origin, window_id, cfg)
                before = predict_window(*args, release=release)
                for value in [-9e8, np.nan]:
                    helper.poison(root, "train", value, origin)
                    fresh = EvaluationRelease(root)
                    fresh.train_grid(panel, "2031-01-02")
                    after = predict_window(*args, release=fresh)
                    pd.testing.assert_frame_equal(before, after, check_exact=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
