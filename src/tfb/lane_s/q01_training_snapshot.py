"""Q-01 selectors, observed labels and the sole Task 2 feature accessor.

No horizon labels enter OnlineView. Official IDs are join keys only. Generated
windows use raw released train observations in [T-60,T-5], masked data through T.
The evaluation functions intentionally read future *train labels*, never features.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Iterator
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TASK2_PANELS = (
    "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S",
)
VALUES = ("speed_kmh", "flow_vph", "occupancy")
DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})\.parquet$")


def project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for root in (current, *current.parents):
        marker = root / ".git"
        if marker.is_dir():
            return root
        if marker.is_file():
            gitdir = Path(marker.read_text(encoding="utf-8").partition(":")[2].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
            return gitdir.parent.parent.parent
    raise FileNotFoundError("Pass release_root outside a Git checkout")


def release_path(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else project_root() / "data/raw/kaggle_public"


def timestamp(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="raise")


def iso(value) -> str:
    return timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_origin(date: str, origin) -> pd.Timestamp:
    t = timestamp(origin)
    if t.strftime("%Y-%m-%d") != str(date):
        raise ValueError("Origin must lie on the requested date (release uses UTC)")
    if t.minute % 5 or t.second or t.microsecond or t.nanosecond:
        raise ValueError("Origin must be on the five-minute grid")
    return t


@dataclass(frozen=True)
class Network:
    panel: str
    links: tuple[str, ...]
    free_speed: np.ndarray
    topology: pd.DataFrame

    @property
    def cutoffs(self) -> np.ndarray:
        return self.free_speed * 0.6

    @property
    def cutoff_by_link(self) -> dict[str, float]:
        return dict(zip(self.links, self.cutoffs))


@dataclass
class DayGrid:
    panel: str
    date: str
    links: tuple[str, ...]
    speed: np.ndarray
    observed: np.ndarray
    eligible: np.ndarray
    queued: np.ndarray


@lru_cache(maxsize=10)
def _network(root: str, panel: str) -> Network:
    base = Path(root) / "corridors" / panel / "network"
    topology = pd.read_csv(base / "lwr_mainline_topology.csv")
    topology = topology.sort_values("order_index", kind="stable").reset_index(drop=True)
    if topology.link_id.duplicated().any() or topology.order_index.duplicated().any():
        raise ValueError("Topology must define unique link positions")
    links = tuple(topology.link_id.astype(str))
    fd = pd.read_csv(base / "fd_parameters.csv").set_index("link_id")
    vf = fd.free_speed_kmh.reindex(links).to_numpy(dtype=np.float64)
    if not np.isfinite(vf).all() or (vf <= 0).any():
        raise ValueError("Every topology link requires a positive released FD speed")
    vf.flags.writeable = False
    return Network(panel, links, vf, topology)


class Release:
    """Released files only; one panel-day at a time, Arrow decoding capped at 4."""

    def __init__(self, root: str | Path | None = None):
        self.root = release_path(root)
        self._file_cache = {}
        pa.set_cpu_count(4)
        pa.set_io_thread_count(4)

    def network(self, panel: str) -> Network:
        return _network(str(self.root.resolve()), panel)

    def files(self, panel: str, split: str, source: str) -> dict[str, Path]:
        if split not in {"train", "validation", "private"}:
            raise ValueError("Unknown split")
        if source not in {"mainline_states", "mainline_states_masked", "ramp_states"}:
            raise ValueError("Unknown released source")
        if source == "mainline_states" and split != "train":
            raise ValueError("Unmasked states are released only for train")
        key = (panel, split, source)
        if key in self._file_cache:
            return self._file_cache[key].copy()
        paths = {}
        for path in (self.root / "corridors" / panel / split / source).glob("*/*.parquet"):
            m = DATE_PATTERN.search(path.name)
            if m is None:
                raise ValueError(f"Unrecognised released day filename: {path.name}")
            day = "-".join(m.groups())
            if day in paths:
                raise ValueError(f"Multiple {source} files for {panel}/{split}/{day}")
            paths[day] = path
        self._file_cache[key] = dict(sorted(paths.items()))
        return self._file_cache[key].copy()

    def read_day(self, panel: str, split: str, date: str, source: str,
                 start=None, end=None) -> pd.DataFrame:
        paths = self.files(panel, split, source)
        if date not in paths:
            raise FileNotFoundError(f"No released {panel}/{split}/{source}/{date}")
        path = paths[date]
        schema = pq.read_schema(path)
        filters = []
        for op, bound in ((">=", start), ("<=", end)):
            if bound is not None:
                value = iso(bound) if (pa.types.is_string(schema.field("timestamp").type) or pa.types.is_large_string(schema.field("timestamp").type)) else timestamp(bound).to_pydatetime()
                filters.append(("timestamp", op, value))
        columns = [c for c in ("timestamp", "link_id", "ramp_link_id", *VALUES, "pct_observed", "is_score_eligible") if c in schema.names]
        frame = pq.read_table(path, columns=columns, filters=filters or None, partitioning=None).to_pandas()
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        if not frame.timestamp.dt.normalize().eq(timestamp(date)).all():
            raise AssertionError("Daily file contains values from another date")
        for col in (*VALUES, "pct_observed", "is_score_eligible"):
            if col in frame:
                frame[col] = frame[col].astype(np.float32)
        return frame.sort_values(["timestamp", "link_id" if "link_id" in frame else "ramp_link_id"]).reset_index(drop=True)

    def window_index(self, panel: str, split: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "task2" / panel / split / "window_index.csv")

    def history(self, panel: str, split: str, date: str, origin) -> pd.DataFrame | None:
        t = check_origin(date, origin)
        index_path = self.root / "task2" / panel / split / "window_index.csv"
        if not index_path.exists():
            return None
        # Only semantic date/origin keys select the history. Condition, coverage,
        # window numbering and row order never enter a feature.
        index = pd.read_csv(index_path, usecols=["window_id", "date", "forecast_origin"])
        match = index[index.date.eq(date) & pd.to_datetime(index.forecast_origin, utc=True).eq(t)]
        if match.empty:
            return None
        if len(match) != 1:
            raise ValueError("Origin must identify exactly one official window")
        path = index_path.with_name("window_history.parquet")
        schema = pq.read_schema(path)
        columns = ["timestamp", "link_id", *VALUES, "pct_observed", "is_score_eligible"]
        columns = [c for c in columns if c in schema.names]
        filters = [("window_id", "==", str(match.iloc[0].window_id)),
                   ("timestamp", ">=", iso(t - pd.Timedelta(minutes=60))),
                   ("timestamp", "<=", iso(t - pd.Timedelta(minutes=5)))]
        frame = pq.read_table(path, columns=columns, filters=filters, partitioning=None).to_pandas()
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        for col in (*VALUES, "pct_observed", "is_score_eligible"):
            if col in frame:
                frame[col] = frame[col].astype(np.float32)
        return frame.sort_values(["timestamp", "link_id"]).reset_index(drop=True)

    def train_grid(self, panel: str, date: str) -> DayGrid:
        """Evaluation/training-label path; NEVER call this from a predictor."""
        return grid_from_frame(self.read_day(panel, "train", date, "mainline_states"), self.network(panel), date)


def grid_from_frame(frame: pd.DataFrame, network: Network, date: str) -> DayGrid:
    times = pd.to_datetime(frame.timestamp, utc=True)
    if not times.dt.normalize().eq(timestamp(date)).all():
        raise ValueError("Grid must be confined to its date")
    steps = (times.dt.hour * 12 + times.dt.minute // 5).to_numpy()
    indices = pd.Index(network.links).get_indexer(frame.link_id.astype(str))
    if (indices < 0).any() or frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError("State cells must be unique and on the topology")
    speed = np.full((288, len(network.links)), np.nan, dtype=np.float32)
    eligible = np.zeros_like(speed, dtype=bool)
    speed[steps, indices] = frame.speed_kmh.to_numpy(dtype=np.float32)
    eligible[steps, indices] = frame.is_score_eligible.fillna(0).astype(bool)
    observed = np.isfinite(speed)
    queued = observed & (speed <= network.cutoffs[None, :])
    return DayGrid(network.panel, date, network.links, speed, observed, eligible, queued)


def candidate_windows(day: DayGrid, *, eligible_only: bool = False,
                      minimum_history_coverage: float = 0.7) -> pd.DataFrame:
    """Offline selector on released train labels, not an online feature."""
    queue = day.queued & day.eligible if eligible_only else day.queued
    cumulative = np.vstack((np.zeros((1, queue.shape[1]), dtype=np.int32), np.cumsum(queue, axis=0, dtype=np.int32)))
    rows = []
    for t in range(12, 282):
        coverage = float(day.eligible[t - 12:t].mean())
        if coverage < minimum_history_coverage:
            continue
        ongoing = bool(((cumulative[t] - cumulative[t - 12]) >= 2).any())
        future = queue[t + 1:t + 7]
        if ongoing and future.any():
            condition = "queue_ongoing"
        elif not ongoing and not future[:5].any() and future[5].any():
            condition = "queue_onset"
        else:
            continue
        valid = day.observed[t + 1:t + 7]
        if eligible_only:
            valid = valid & day.eligible[t + 1:t + 7]
        persistence = np.broadcast_to(queue[t - 1], future.shape) & valid
        intersection = int((persistence & future).sum())
        union = int((persistence | future).sum())
        proxy_iou = intersection / union if union else 1.0
        origin = timestamp(day.date) + pd.Timedelta(minutes=t * 5)
        rows.append({"panel": day.panel, "split": "train", "date": day.date,
                     "forecast_origin": iso(origin), "condition": condition,
                     "history_start": iso(origin - pd.Timedelta(minutes=60)),
                     "history_end": iso(origin - pd.Timedelta(minutes=5)),
                     "forecast_start": iso(origin + pd.Timedelta(minutes=5)),
                     "forecast_end": iso(origin + pd.Timedelta(minutes=30)),
                     "history_coverage": coverage, "history_last_step_persistence_proxy_iou": proxy_iou})
    return pd.DataFrame(rows, columns=["panel", "split", "date", "forecast_origin", "condition", "history_start", "history_end", "forecast_start", "forecast_end", "history_coverage", "history_last_step_persistence_proxy_iou"])


def horizon_labels(day: DayGrid, origin, *, eligible_only: bool = False) -> pd.DataFrame:
    """Observed-speed proxy labels; null cells excluded unless explicitly filled.

    This is evaluation-only full-day interpolation. It must not enter OnlineView.
    Raw noise-free queue truth is withheld; these are not organizer ground truth.
    """
    t = check_origin(day.date, origin)
    step = t.hour * 12 + t.minute // 5
    if step + 6 >= 288:
        raise ValueError("Horizon crosses the daily grid")
    observed = day.observed[step + 1:step + 7]
    queue = day.queued[step + 1:step + 7]
    valid = observed & day.eligible[step + 1:step + 7] if eligible_only else observed
    n = len(day.links)
    return pd.DataFrame({"timestamp": np.repeat(pd.date_range(t + pd.Timedelta(minutes=5), periods=6, freq="5min"), n),
                         "link_id": np.tile(day.links, 6), "queue_true": queue.reshape(-1).astype(np.int8),
                         "is_score_eligible": valid.reshape(-1).astype(np.int8),
                         "label_observed": observed.reshape(-1).astype(np.int8)})


def interpolated_labels(day: DayGrid, origin, network: Network, *, eligible_only: bool = False) -> pd.DataFrame:
    filled = pd.DataFrame(day.speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    grid = DayGrid(day.panel, day.date, day.links, filled, np.isfinite(filled), day.eligible,
                   np.isfinite(filled) & (filled <= network.cutoffs[None, :]))
    labels = horizon_labels(grid, origin, eligible_only=eligible_only)
    t = check_origin(day.date, origin)
    step = t.hour * 12 + t.minute // 5
    labels["label_observed"] = day.observed[step + 1:step + 7].reshape(-1).astype(np.int8)
    return labels


class OnlineView:
    """Allowed feature values and explicit archives. No future-label attribute.

    Every same-split value is timestamp <= T; same-day raw train values are
    restricted to the 12 history steps. All train is available for references on
    validation/private; train examples get only prior dates to preserve causality.
    """

    def __init__(self, release: Release, panel: str, split: str, date: str, origin):
        self._release, self.panel, self.split, self.date = release, panel, split, date
        self.origin = check_origin(date, origin)
        self.network = release.network(panel)

    @cached_property
    def _history(self):
        history = self._release.history(self.panel, self.split, self.date, self.origin)
        if history is None:
            if self.split != "train":
                raise ValueError("Unreleased validation/private window history")
            history = self._release.read_day(self.panel, "train", self.date, "mainline_states",
                                            self.origin - pd.Timedelta(minutes=60),
                                            self.origin - pd.Timedelta(minutes=5))
        return self._safe(history, start=self.origin - pd.Timedelta(minutes=60),
                          end=self.origin - pd.Timedelta(minutes=5))

    @cached_property
    def _mainline(self):
        return self._safe(self._release.read_day(self.panel, self.split, self.date,
                                                "mainline_states_masked", end=self.origin))

    @cached_property
    def _ramps(self):
        return self._safe(self._release.read_day(self.panel, self.split, self.date,
                                                "ramp_states", end=self.origin))

    def _safe(self, frame, start=None, end=None):
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        if frame.timestamp.isna().any():
            raise AssertionError("Missing timestamp in feature input")
        # Filter again for custom loaders and fail closed on a bound violation.
        frame = frame[frame.timestamp.le(self.origin)]
        if start is not None:
            frame = frame[frame.timestamp.ge(start)]
        if end is not None:
            frame = frame[frame.timestamp.le(end)]
        if not frame.empty and (frame.timestamp.max() > self.origin or
                                not frame.timestamp.dt.normalize().eq(timestamp(self.date)).all()):
            raise AssertionError("Online accessor crossed the origin/date")
        # No IDs, ordering, missingness flags, regime or coverage metadata.
        columns = [c for c in ("timestamp", "link_id", "ramp_link_id", *VALUES, "pct_observed", "is_score_eligible") if c in frame]
        frame = frame[columns]
        key = "link_id" if "link_id" in frame else "ramp_link_id"
        return frame.sort_values(["timestamp", key]).reset_index(drop=True)

    @property
    def history(self) -> pd.DataFrame:
        return self._history.copy()

    @property
    def mainline(self) -> pd.DataFrame:
        return self._mainline.copy()

    @property
    def ramps(self) -> pd.DataFrame:
        return self._ramps.copy()

    def observations(self) -> pd.DataFrame:
        """Same-day masked values with raw history overriding its 12 steps."""
        keys = ["timestamp", "link_id"]
        out = pd.concat([self._mainline, self._history], ignore_index=True)
        out = out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
        assert out.empty or out.timestamp.max() <= self.origin
        return out

    def condition(self, *, eligible_only: bool = False) -> str:
        h = self._history
        queued = h.speed_kmh.notna() & h.speed_kmh.le(h.link_id.map(self.network.cutoff_by_link))
        if eligible_only:
            queued &= h.is_score_eligible.fillna(0).astype(bool)
        counts = queued.groupby(h.link_id).sum()
        return "queue_ongoing" if counts.ge(2).any() else "queue_onset"

    def previous_days(self, source="mainline_states_masked", start_date=None) -> Iterator[tuple[str, pd.DataFrame]]:
        for date in self._release.files(self.panel, self.split, source):
            if date >= self.date or (start_date is not None and date < start_date):
                continue
            frame = self._release.read_day(self.panel, self.split, date, source)
            if not frame.empty and frame.timestamp.max() >= timestamp(self.date):
                raise AssertionError("Prior-date archive crossed current date")
            yield date, frame

    def training_reference(self, start_date=None, end_date=None) -> Iterator[tuple[str, pd.DataFrame]]:
        """All released train on held splits; earlier dates on train examples."""
        for date in self._release.files(self.panel, "train", "mainline_states"):
            if (start_date is not None and date < start_date) or (end_date is not None and date > end_date):
                continue
            if self.split == "train" and date >= self.date:
                continue
            yield date, self._release.read_day(self.panel, "train", date, "mainline_states")


def online_view(panel: str, split: str, date: str, T, *, release_root=None,
                release: Release | None = None) -> OnlineView:
    return OnlineView(release or Release(release_root), panel, split, date, T)


def frozen_windows(candidates: pd.DataFrame, *, per_condition: int = 5,
                   seed: int = 106, first_onset_per_day: bool = True) -> pd.DataFrame:
    """First onset per day (or chronological 360-minute spacing); ongoing randomly chosen on calendar days 1..12.

    Random ongoing choices respect the observed 07:30..19:30 range and at least
    360 minutes within each condition. No fitted forecasts or ADV scores select
    these lists. Ordering is explicit semantic time, never released row order.
    """
    if candidates.empty:
        return candidates.copy()
    work = candidates.copy()
    work["_t"] = pd.to_datetime(work.forecast_origin, utc=True)
    work["_month"] = work._t.dt.strftime("%Y-%m")
    rng = np.random.default_rng(seed)
    selected = []
    for (panel, month), group in work.groupby(["panel", "_month"], sort=True):
        onset = group[group.condition.eq("queue_onset")].sort_values("_t")
        if first_onset_per_day:
            onset = onset.drop_duplicates("date").head(per_condition)
        else:
            kept_onsets = []
            for _, row in onset.iterrows():
                if not kept_onsets or row._t - kept_onsets[-1]._t >= pd.Timedelta(minutes=360):
                    kept_onsets.append(row)
                if len(kept_onsets) == per_condition:
                    break
            onset = pd.DataFrame(kept_onsets)
        ongoing = group[group.condition.eq("queue_ongoing")].sort_values("_t")
        if "history_last_step_persistence_proxy_iou" in ongoing:
            ongoing = ongoing[ongoing.history_last_step_persistence_proxy_iou.le(0.9)]
        minute = ongoing._t.dt.hour * 60 + ongoing._t.dt.minute
        ongoing = ongoing[ongoing._t.dt.day.le(12) & minute.between(450, 1170)]
        kept = []
        for i in rng.permutation(len(ongoing)):
            row = ongoing.iloc[int(i)]
            if all(abs(row._t - prior._t) >= pd.Timedelta(minutes=360) for prior in kept):
                kept.append(row)
            if len(kept) == per_condition:
                break
        if len(onset) != per_condition or len(kept) != per_condition:
            raise ValueError(f"Insufficient frozen candidates for {panel}/{month}")
        selected.append(onset)
        selected.append(pd.DataFrame(kept))
    result = pd.concat(selected, ignore_index=True).drop(columns=["_t", "_month"])
    result = result.sort_values(["panel", "forecast_origin", "condition"]).reset_index(drop=True)
    result["family_id"] = result.panel.str.rsplit("_", n=1).str[0]
    # These IDs are synthetic output joins and must never be used as features.
    result["window_id"] = [f"sim_{p}_{iso(t)}" for p, t in zip(result.panel, result.forecast_origin)]
    origins = pd.to_datetime(result.forecast_origin, utc=True)
    result["gap_start"] = [iso(t + pd.Timedelta(minutes=5)) for t in origins]
    result["gap_end"] = [iso(t + pd.Timedelta(minutes=90)) for t in origins]
    result["instrument"] = "I-Q-ADV frozen / I-S-ADV gap windows"
    return result


def adverse_population(candidates: pd.DataFrame, *, ongoing_per_day: int = 20,
                       seed: int = 106) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = []
    for _, group in candidates.groupby(["panel", "date"], sort=True):
        chunks.append(group[group.condition.eq("queue_onset")])
        ongoing = group[group.condition.eq("queue_ongoing")].sort_values("forecast_origin")
        choice = rng.choice(len(ongoing), min(ongoing_per_day, len(ongoing)), replace=False)
        chunks.append(ongoing.iloc[choice])
    return pd.concat(chunks, ignore_index=True).sort_values(["panel", "forecast_origin"]).reset_index(drop=True)
