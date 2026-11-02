"""Timing metric primitives.

All functions operate on parallel lists of words and ``(start, end)`` timings in
seconds. Two families of metrics are provided:

* MAE / median AE / mIoU / median IoU on text-aligned word pairs
  (transcription-decoupled).
* F1 at a collar tolerance, in two variants:
    - standard segmentation F1 (word identity + boundary tolerance), and
    - matched-word collar accuracy (fraction of already text-aligned pairs with
      both boundaries inside the collar; stored as ``f1_matched`` for backward
      compatibility, but it is an accuracy, not an F1).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .normalize import normalize_word

Timing = Tuple[Optional[float], Optional[float]]


def iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Intersection-over-union of two time intervals."""
    inter = max(0.0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return float(inter / union) if union > 0 else 0.0


def _valid(t: Timing) -> bool:
    return t is not None and t[0] is not None and t[1] is not None


def mae_on_aligned(
    pred_times: Sequence[Timing],
    gt_times: Sequence[Timing],
    alignment: Sequence[Tuple[int, int]],
) -> dict:
    """Per-boundary absolute errors and IoU over aligned pairs.

    Returns raw error lists (seconds) so callers can aggregate micro/macro.
    """
    start_errors: List[float] = []
    end_errors: List[float] = []
    start_signed: List[float] = []
    end_signed: List[float] = []
    ious: List[float] = []

    for pi, gi in alignment:
        if pi >= len(pred_times) or gi >= len(gt_times):
            continue
        ps, pe = pred_times[pi]
        gs, ge = gt_times[gi]
        if not _valid((ps, pe)) or not _valid((gs, ge)):
            continue
        start_errors.append(abs(ps - gs))
        end_errors.append(abs(pe - ge))
        start_signed.append(ps - gs)
        end_signed.append(pe - ge)
        ious.append(iou(ps, pe, gs, ge))

    return {
        "start_errors": start_errors,
        "end_errors": end_errors,
        "start_signed": start_signed,
        "end_signed": end_signed,
        "ious": ious,
    }


def standard_segmentation_counts(
    pred_words: Sequence[str],
    pred_times: Sequence[Timing],
    gt_words: Sequence[str],
    gt_times: Sequence[Timing],
    collar: float,
) -> Tuple[int, int, int]:
    """Transcription-coupled segmentation counts ``(tp, fp, fn)``.

    A predicted word is a true positive if there exists an unmatched ground-truth
    word with the same normalized text whose start and end are both within
    ``collar`` seconds.
    """
    gw = [normalize_word(w) for w in gt_words]
    pw = [normalize_word(w) for w in pred_words]

    gt_matched = [False] * len(gw)
    tp = 0
    for i, (ps, pe) in enumerate(pred_times):
        if not _valid((ps, pe)) or not pw[i]:
            continue
        for j, (gs, ge) in enumerate(gt_times):
            if gt_matched[j] or not gw[j] or not _valid((gs, ge)):
                continue
            if gw[j] == pw[i] and abs(ps - gs) <= collar and abs(pe - ge) <= collar:
                gt_matched[j] = True
                tp += 1
                break

    num_pred = sum(1 for i, t in enumerate(pred_times) if _valid(t) and pw[i])
    num_gt = sum(1 for j, t in enumerate(gt_times) if _valid(t) and gw[j])
    fp = num_pred - tp
    fn = num_gt - tp
    return tp, fp, fn


def matched_f1_counts(
    pred_times: Sequence[Timing],
    gt_times: Sequence[Timing],
    alignment: Sequence[Tuple[int, int]],
    collar: float,
) -> Tuple[int, int]:
    """Matched-word timing counts ``(tp, count)``.

    Among already text-aligned pairs, a pair is a true positive if both
    boundaries are within ``collar`` seconds. ``count`` is the number of valid
    aligned pairs considered.
    """
    tp = 0
    count = 0
    for pi, gi in alignment:
        if pi >= len(pred_times) or gi >= len(gt_times):
            continue
        ps, pe = pred_times[pi]
        gs, ge = gt_times[gi]
        if not _valid((ps, pe)) or not _valid((gs, ge)):
            continue
        count += 1
        if abs(ps - gs) <= collar and abs(pe - ge) <= collar:
            tp += 1
    return tp, count


def summarize(values: Sequence[float], scale: float = 1.0) -> dict:
    """Mean / median / p90 of ``values`` multiplied by ``scale`` (None if empty)."""
    if not values:
        return {"mean": None, "median": None, "p90": None}
    arr = np.asarray(values, dtype=float) * scale
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    """Precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def collar_label(collar_s: float) -> str:
    """Human-readable collar label, e.g. 0.05 -> ``50ms``."""
    ms = collar_s * 1000.0
    if abs(ms - round(ms)) < 1e-6:
        return f"{int(round(ms))}ms"
    return f"{ms:g}ms"
