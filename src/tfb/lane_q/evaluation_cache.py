"""One-panel-day evaluation cache. Features remain bounded through OnlineView.

Released full train labels must be read by evaluation. This cache shares that
read, but returns only each requested temporal slice to the online accessor.
Production predict_queue keeps using the Parquet-filtered Release by default.
"""
from __future__ import annotations

from .windows import Release, timestamp


class EvaluationRelease(Release):
    def __init__(self, root=None):
        super().__init__(root)
        self._active = None
        self._day_frames = {}

    def read_day(self, panel, split, date, source, start=None, end=None):
        if split != "train":
            return super().read_day(panel, split, date, source, start, end)
        active = (panel, split, date)
        if active != self._active:
            self._active = active
            self._day_frames.clear()
        if source not in self._day_frames:
            self._day_frames[source] = super().read_day(panel, split, date, source)
        frame = self._day_frames[source]
        keep = frame.timestamp.notna()
        if start is not None:
            keep &= frame.timestamp.ge(timestamp(start))
        if end is not None:
            keep &= frame.timestamp.le(timestamp(end))
        return frame.loc[keep].copy().reset_index(drop=True)
