"""Canonical Q-02 baseline. Every prediction feature comes from OnlineView."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .windows import Release, iso, online_view, timestamp


MODEL_VERSION = 1
KEYS = ["window_id", "timestamp", "link_id"]


def calendar_features(view):
    target = view.origin + pd.Timedelta(minutes=30)
    return int(target.hour), "weekend" if target.weekday() >= 5 else "weekday"


def last_queue(view):
    """Last finite speed per link, including allowed masked observations at T."""
    observations = view.observations()
    valid = observations[np.isfinite(observations.speed_kmh)]
    last = valid.sort_values(["timestamp", "link_id"]).groupby("link_id", sort=False).tail(1)
    speed = last.set_index("link_id").speed_kmh.reindex(view.network.links).to_numpy(dtype=np.float32)
    return np.isfinite(speed) & (speed <= view.network.cutoffs)


def fit_panel_prior(records, links, *, max_position_gap=2):
    """Cluster co-occurring nearby cells, then count site and cell-set modes.

    No IDs or row order are predictors. Ties use physical topology positions.
    Sites may remain distinct even when adjacent if they never co-occur.
    """
    positions = {link: i for i, link in enumerate(links)}
    parent = {}
    def find(i):
        parent.setdefault(i, i)
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    prepared = []
    for record in records:
        cells = tuple(sorted({positions[link] for link in record["target_links"]}))
        if not cells:
            continue
        for i in cells:
            find(i)
        for a, b in zip(cells, cells[1:]):
            if b - a <= max_position_gap:
                union(a, b)
        prepared.append((int(record["hour"]), record["day_type"], cells))
    groups = defaultdict(list)
    for i in sorted(parent):
        groups[find(i)].append(i)
    site_positions = sorted(groups.values(), key=lambda group: tuple(group))
    site_of = {i: s for s, cells in enumerate(site_positions) for i in cells}
    counters = defaultdict(Counter)
    sets = defaultdict(lambda: defaultdict(Counter))
    all_sets = defaultdict(Counter)
    for hour, day_type, cells in prepared:
        by_site = defaultdict(list)
        for i in cells:
            by_site[site_of[i]].append(i)
        contexts = [f"hour_day:{hour}/{day_type}", f"hour:{hour}", f"day:{day_type}", "all"]
        for site, subset in by_site.items():
            cell_set = tuple(subset)
            all_sets[site][cell_set] += 1
            for context in contexts:
                counters[context][site] += 1
                sets[context][site][cell_set] += 1
    context_model = {}
    for context in sorted(counters):
        counts = counters[context]
        site = min(counts, key=lambda s: (-counts[s], tuple(site_positions[s])))
        distribution = sets[context][site]
        modal = min(distribution, key=lambda cells: (-distribution[cells], cells))
        context_model[context] = {"site": site, "target_links": [links[i] for i in modal],
                                  "n_site_occurrences": int(counts[site]),
                                  "site_counts": {str(s): int(n) for s, n in sorted(counts.items())}}
    sites = [{"site": site, "positions": cells, "links": [links[i] for i in cells],
              "n_windows": int(sum(all_sets[site].values())),
              "cell_sets": [{"links": [links[i] for i in cell_set], "count": int(count)}
                            for cell_set, count in sorted(all_sets[site].items(), key=lambda item: (-item[1], item[0]))]}
             for site, cells in enumerate(site_positions)]
    return {"links": list(links), "n_onsets": len(prepared), "sites": sites, "contexts": context_model}


def choose_onset_links(view, prior):
    model = prior["panels"][view.panel]
    if tuple(model["links"]) != view.network.links:
        raise ValueError("Prior topology does not match the released panel")
    hour, day_type = calendar_features(view)
    for context in [f"hour_day:{hour}/{day_type}", f"hour:{hour}", f"day:{day_type}", "all"]:
        if context in model["contexts"]:
            return tuple(model["contexts"][context]["target_links"])
    return ()


def forecast(view, prior, *, onset_mode="site_prior"):
    """Six-step binary grid. Horizon values and provided condition never enter."""
    result = np.zeros((6, len(view.network.links)), dtype=np.int8)
    condition = view.condition()
    if condition == "queue_ongoing":
        result[:] = last_queue(view)
    elif onset_mode == "site_prior":
        cells = set(choose_onset_links(view, prior))
        result[5] = [int(link in cells) for link in view.network.links]
    elif onset_mode == "zero":
        pass
    else:
        raise ValueError("Unknown onset mode")
    return result


def load_prior(cfg):
    if "prior" in cfg:
        prior = cfg["prior"]
    else:
        path = cfg.get("prior_path")
        if not path and cfg.get("ongoing_model_dir"):
            path=Path(cfg["ongoing_model_dir"])/"prior.json"
        if not path and cfg.get("ongoing_model"):
            return cfg["ongoing_model"].prior
        if not path and cfg.get("onset_model_dir"):
            path = Path(cfg["onset_model_dir"])/"prior.json"
        if not path and cfg.get("onset_model"):
            return cfg["onset_model"].prior
        if not path:
            raise ValueError("Q-02 requires cfg.prior_path (released train-only prior JSON)")
        prior = json.loads(Path(path).read_text(encoding="utf-8"))
    if prior.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported prior model version")
    return prior


def load_onset_model(directory, *, num_threads=None):
    """Dispatch the explicitly versioned feature builder recorded by the model."""
    metadata=json.loads((Path(directory)/"metadata.json").read_text(encoding="utf-8"))
    builder=metadata.get("feature_builder")
    if builder=="Q07_transport_v1":
        from .precursor_model import PrecursorModel
        return PrecursorModel.load(directory,num_threads=num_threads)
    if builder is not None:raise ValueError("Unsupported onset feature builder")
    from .onset_model import OnsetModel
    return OnsetModel.load(directory,num_threads=num_threads)


def load_ongoing_model(directory, *, num_threads=None, release=None):
    """Load the recorded ongoing feature builder without guessing from paths."""
    metadata=json.loads((Path(directory)/"metadata.json").read_text(encoding="utf-8"))
    builder=metadata.get("feature_builder")
    if builder=="C03_C05_online_cnn_v1":
        if metadata.get('condition') != 'queue_ongoing':
            raise ValueError('Ongoing CNN loader requires its recorded ongoing condition')
        from .cnn_production import CNNProduction
        release = release or Release()
        return CNNProduction.load(directory, root=release.root.parents[2], release=release,
                                  num_threads=2 if num_threads is None else num_threads)
    if builder=="Q05_canonical_ctm_v1":
        from .ctm_model import CTMModel
        return CTMModel.load(directory,num_threads=num_threads)
    if builder is not None:raise ValueError("Unsupported ongoing feature builder")
    from .ongoing_runtime import OngoingPredictor
    return OngoingPredictor.load(directory,num_threads=num_threads)


def predict_window(panel, split, date, origin, window_id, cfg, *, release=None):
    release = release or Release(cfg.get("release_root"))
    view = online_view(panel, split, date, origin, release=release)
    prior = load_prior(cfg)
    if view.condition()=="queue_ongoing" and (cfg.get("ongoing_model_dir") or cfg.get("ongoing_model")):
        model=cfg.get("ongoing_model") or load_ongoing_model(cfg["ongoing_model_dir"],num_threads=cfg.get("num_threads"),release=release)
        values=model.forecast(view)
    elif cfg.get("onset_model_dir") or cfg.get("onset_model"):
        model=cfg.get("onset_model") or load_onset_model(cfg["onset_model_dir"],num_threads=cfg.get("num_threads"))
        values=model.forecast(view)
    else:
        values = forecast(view, prior, onset_mode=cfg.get("onset_mode", "site_prior"))
    times = [iso(view.origin + pd.Timedelta(minutes=5 * step)) for step in range(1, 7)]
    return pd.DataFrame({"window_id": str(window_id), "timestamp": np.repeat(times, len(view.network.links)),
                         "link_id": np.tile(view.network.links, 6), "queue_pred": values.reshape(-1)})


def predict_queue(panel, split, cfg):
    """Return one binary prediction for every released template key."""
    cfg = dict(cfg)
    release = Release(cfg.get("release_root"))
    template_path = release.root / "task2" / panel / split / "sample_submission_queue.csv"
    if not template_path.exists():
        if panel in {"D12_I405_N", "D12_I405_S"}:
            return pd.DataFrame({"window_id": pd.Series(dtype=str), "timestamp": pd.Series(dtype=str),
                                 "link_id": pd.Series(dtype=str), "queue_pred": pd.Series(dtype=np.int8)})
        raise FileNotFoundError(template_path)
    template = pd.read_csv(template_path, usecols=KEYS, dtype={"window_id": str, "link_id": str})
    template["timestamp"] = [iso(t) for t in pd.to_datetime(template.timestamp, utc=True)]
    if template.duplicated(KEYS).any():
        raise ValueError("Duplicate template key")
    # Index IDs select released history and output keys, never features.
    index = pd.read_csv(template_path.with_name("window_index.csv"),
                        usecols=["window_id", "date", "forecast_origin"], dtype={"window_id": str})
    prior = load_prior(cfg)
    cfg["prior"] = prior
    # Verify the shared runtime before loading either model.
    if cfg.get("ongoing_model_dir") and not cfg.get("ongoing_model"):
        cfg["ongoing_model"]=load_ongoing_model(cfg["ongoing_model_dir"],num_threads=cfg.get("num_threads"),release=release)
    if cfg.get("onset_model_dir") and not cfg.get("onset_model"):
        cfg["onset_model"]=load_onset_model(cfg["onset_model_dir"],num_threads=cfg.get("num_threads"))
    chunks = [predict_window(panel, split, w.date, w.forecast_origin, w.window_id, cfg, release=release)
              for w in index.itertuples(index=False)]
    prediction = pd.concat(chunks, ignore_index=True)
    result = template.merge(prediction, on=KEYS, how="left", validate="one_to_one")
    if len(result) != len(template) or result.queue_pred.isna().any() or not result.queue_pred.isin([0, 1]).all():
        raise AssertionError("Q-02 predictions must fill exactly the binary template")
    if len(prediction.merge(template, on=KEYS, how="outer", indicator=True).query("_merge != 'both'")):
        raise AssertionError("Forecast grid differs from the released template")
    result["queue_pred"] = result.queue_pred.astype(np.int8)
    return result
