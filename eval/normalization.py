# -*- coding: utf-8 -*-
"""
Ground-truth normalization utilities for Soccer-GMR evaluation.

Supported annotation formats:
  - relevant_windows: [[st, ed], ...]
  - raw moment structures with moment.type == "clips" or "timestamps";
    timestamp annotations require ts_cfg to expand points into windows.

This module only normalizes structure and validity. It does not compute metrics.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def sanitize_windows(
    windows: Optional[List],
    duration: Optional[float] = None,
) -> List[List[float]]:
    """
    Clean GT windows into valid, deduplicated [st, ed] windows sorted by start time.

    Args:
        windows: Raw window list.
        duration: If given, clip boundaries to [0, duration]; skip clipping when unknown.
    """
    cleaned: List[List[float]] = []
    for w in windows or []:
        if w is None or len(w) != 2:
            continue
        try:
            st, ed = float(w[0]), float(w[1])
        except (TypeError, ValueError):
            continue
        if duration is not None:
            try:
                dur = float(duration)
                st, ed = max(0.0, st), min(dur, ed)
            except (TypeError, ValueError):
                pass
        if ed <= st:
            continue
        cleaned.append([st, ed])
    cleaned.sort(key=lambda x: (x[0], x[1]))
    deduped: List[List[float]] = []
    last: Optional[List[float]] = None
    for w in cleaned:
        if last is None or w[0] != last[0] or w[1] != last[1]:
            deduped.append(w)
        last = w
    return deduped


def load_ts_window_cfg(cfg_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load timestamp expansion rules from a JSON file; return None when no path is given."""
    if cfg_path is None:
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"ts_window_cfg must be a JSON object, got: {type(cfg)}")
    return cfg


def get_pre_post_by_query_type(
    ts_cfg: Optional[Dict[str, Any]],
    query_type: Any,
) -> Tuple[float, float]:
    """
    Resolve timestamp expansion seconds before and after the point by query_type.
    Fall back to default when by_query_type has no match; raise if no rule exists.
    """
    if ts_cfg is None:
        raise ValueError("GT uses timestamp moments, but ts_cfg was not provided. Pass --gt_ts_window_cfg.")
    default = ts_cfg.get("default", None)
    by_qt = ts_cfg.get("by_query_type", {}) or {}
    rule = by_qt.get(str(query_type), None) if query_type is not None else None
    if rule is None:
        rule = default
    if rule is None:
        raise ValueError(f"missing ts window rule for query_type={query_type}")
    return float(rule.get("pre", 6.0)), float(rule.get("post", 2.0))


def gt_record_to_relevant_windows(
    d: Dict[str, Any],
    ts_cfg: Optional[Dict[str, Any]],
) -> List[List[float]]:
    """
    Parse one GT record into relevant_windows as second-level [st, ed] windows.

    Prefer top-level relevant_windows when present, including records that also have moment.
    """
    if "relevant_windows" in d:
        raw = d["relevant_windows"]
        return raw if isinstance(raw, list) else []

    moment = d.get("moment") or {}
    mtype = moment.get("type", None)
    value = moment.get("value", None)
    if mtype is None:
        return []

    if mtype == "clips":
        if value is None:
            return []
        # A single window may be stored as a pair instead of a list of pairs.
        if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(
            value[0], (int, float)
        ):
            return [list(value)]
        return list(value or [])

    if mtype == "timestamps":
        query_type = d.get("query_type", None)
        pre, post = get_pre_post_by_query_type(ts_cfg, query_type)
        windows: List[List[float]] = []
        for t in value or []:
            try:
                windows.append([float(t) - pre, float(t) + post])
            except (TypeError, ValueError):
                continue
        return windows

    raise ValueError(f"Unknown moment.type={mtype}")


def normalize_ground_truth(
    gt_raw: List[Dict[str, Any]],
    ts_cfg: Optional[Dict[str, Any]],
    *,
    drop_empty_gt: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Normalize raw GT JSONL records into {qid, relevant_windows} items.

    Args:
        gt_raw: Raw records.
        ts_cfg: Timestamp window config, matching the return value of load_ts_window_cfg.
        drop_empty_gt: If True, drop samples with no positive windows. Main GMR evaluation
            usually keeps empty sets, so callers pass False.
    """
    stats = {"total": len(gt_raw), "kept": 0, "dropped_empty": 0, "dropped_invalid": 0}
    normalized: List[Dict[str, Any]] = []

    for d in gt_raw:
        if not isinstance(d, dict) or "qid" not in d:
            stats["dropped_invalid"] += 1
            continue
        windows = gt_record_to_relevant_windows(d, ts_cfg)
        windows = sanitize_windows(windows, duration=d.get("duration"))
        if drop_empty_gt and len(windows) == 0:
            stats["dropped_empty"] += 1
            continue
        normalized.append({"qid": d["qid"], "relevant_windows": windows})
        stats["kept"] += 1

    return normalized, stats
