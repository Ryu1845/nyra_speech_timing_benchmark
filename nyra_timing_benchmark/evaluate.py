"""Evaluation orchestration: align predictions to ground truth and aggregate.

The same scoring code path is used whether predictions come from a file or from a
model adapter. Metrics are aggregated two ways:

* micro: pool raw errors / counts across all samples (headline numbers), and
* macro: average per-sample values (less sensitive to long utterances).

Protocol (one matching for every metric):

* Scoring set: lexical words and fillers; non-speech events (laughter, bracketed
  tags) are removed from reference and prediction before matching.
* One time-mediated edit-distance alignment per sample; MAE, medians, IoU,
  collar accuracy and F1 are all computed on its matched pairs.
* The unit is the **word**: each matched word has one error
  ``(|d_start| + |d_end|) / 2`` and one IoU. MAE / MedAE / P90 are over these
  per-word errors (start/end are also reported separately). A hit at collar
  ``t`` needs both boundaries within ``t``; F1 counts hits against all predicted
  (precision) and all reference (recall) words, so missed words cost recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import align_words
from .metrics import (
    collar_label,
    mae_on_aligned,
    matched_f1_counts,
    prf,
    summarize,
)
from .normalize import is_scored_word

DEFAULT_COLLARS: Tuple[float, ...] = (0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

Timing = Tuple[float, float]


@dataclass
class SampleResult:
    key: str
    num_pred_words: int
    num_gt_words: int
    num_aligned: int
    num_gt_excluded: int = 0
    start_errors: List[float] = field(default_factory=list)
    end_errors: List[float] = field(default_factory=list)
    start_signed: List[float] = field(default_factory=list)
    end_signed: List[float] = field(default_factory=list)
    ious: List[float] = field(default_factory=list)
    # collar -> (tp, fp, fn) standard ; (tp, count) matched
    standard_counts: Dict[float, Tuple[int, int, int]] = field(default_factory=dict)
    matched_counts: Dict[float, Tuple[int, int]] = field(default_factory=dict)

    @property
    def word_errors(self) -> List[float]:
        return [(a + b) / 2.0 for a, b in zip(self.start_errors, self.end_errors)]

    @property
    def overall_mae_ms(self) -> Optional[float]:
        errs = self.word_errors
        return float(np.mean(errs) * 1000.0) if errs else None

    @property
    def median_ae_ms(self) -> Optional[float]:
        errs = self.word_errors
        return float(np.median(errs) * 1000.0) if errs else None

    @property
    def miou(self) -> Optional[float]:
        return float(np.mean(self.ious)) if self.ious else None

    @property
    def median_iou(self) -> Optional[float]:
        return float(np.median(self.ious)) if self.ious else None


def evaluate_sample(
    pred_words: Sequence[str],
    pred_times: Sequence[Timing],
    gt_words: Sequence[str],
    gt_times: Sequence[Timing],
    key: str = "",
    collars: Sequence[float] = DEFAULT_COLLARS,
) -> SampleResult:
    """Align one prediction/GT pair and compute its raw metric accumulators."""
    pk = [i for i, w in enumerate(pred_words) if is_scored_word(w) and _valid(pred_times[i])]
    gk = [j for j, w in enumerate(gt_words) if is_scored_word(w) and _valid(gt_times[j])]
    num_gt_excluded = len(gt_words) - len(gk)
    pred_words = [pred_words[i] for i in pk]
    pred_times = [pred_times[i] for i in pk]
    gt_words = [gt_words[j] for j in gk]
    gt_times = [gt_times[j] for j in gk]

    alignment = align_words(pred_words, gt_words, pred_times, gt_times)
    errs = mae_on_aligned(pred_times, gt_times, alignment)

    result = SampleResult(
        key=key,
        num_pred_words=len(pred_words),
        num_gt_words=len(gt_words),
        num_aligned=len(alignment),
        num_gt_excluded=num_gt_excluded,
        start_errors=errs["start_errors"],
        end_errors=errs["end_errors"],
        start_signed=errs["start_signed"],
        end_signed=errs["end_signed"],
        ious=errs["ious"],
    )
    for c in collars:
        tp, count = matched_f1_counts(pred_times, gt_times, alignment, c)
        result.matched_counts[c] = (tp, count)
        result.standard_counts[c] = (tp, len(pred_words) - tp, len(gt_words) - tp)
    return result


def _valid(t) -> bool:
    return t is not None and t[0] is not None and t[1] is not None


@dataclass
class BenchmarkResult:
    model: str
    dataset: Optional[str]
    collars: List[float]
    overall: Dict = field(default_factory=dict)
    per_sample: List[Dict] = field(default_factory=list)
    condition: Dict = field(default_factory=lambda: {"noise": "clean"})
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "condition": self.condition,
            "metadata": self.metadata,
            "collars": self.collars,
            "overall": self.overall,
            "per_sample": self.per_sample,
        }


def _mean_ms(values: List[float]) -> Optional[float]:
    return float(np.mean(values) * 1000.0) if values else None


def evaluate_dataset(
    ground_truth: Dict[str, "object"],
    predictions: Dict[str, Tuple[Sequence[str], Sequence[Timing]]],
    model: str = "model",
    dataset: Optional[str] = None,
    collars: Sequence[float] = DEFAULT_COLLARS,
    condition: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> BenchmarkResult:
    """Score all ground-truth samples and aggregate.

    ``ground_truth`` maps sample key -> object with ``.words`` and ``.timings``.
    ``predictions`` maps sample key -> ``(words, timings)``. Ground-truth samples
    without a prediction are scored as empty (counted in coverage).
    """
    collars = list(collars)
    sample_results: List[SampleResult] = []

    for key, gt in ground_truth.items():
        pred_words, pred_times = predictions.get(key, ([], []))
        sr = evaluate_sample(
            pred_words, pred_times, gt.words, gt.timings, key=key, collars=collars
        )
        sample_results.append(sr)

    # --- micro (pooled) accumulators ---
    all_start_err: List[float] = []
    all_end_err: List[float] = []
    all_start_signed: List[float] = []
    all_end_signed: List[float] = []
    all_ious: List[float] = []
    std_counts = {c: [0, 0, 0] for c in collars}
    matched_counts = {c: [0, 0] for c in collars}

    n_pred_words = n_gt_words = n_aligned = n_gt_excluded = 0
    n_scored = 0  # samples with at least one prediction word

    per_sample: List[Dict] = []
    macro_mae: List[float] = []
    macro_miou: List[float] = []

    for sr in sample_results:
        all_start_err += sr.start_errors
        all_end_err += sr.end_errors
        all_start_signed += sr.start_signed
        all_end_signed += sr.end_signed
        all_ious += sr.ious
        n_pred_words += sr.num_pred_words
        n_gt_words += sr.num_gt_words
        n_gt_excluded += sr.num_gt_excluded
        n_aligned += sr.num_aligned
        if sr.num_pred_words > 0:
            n_scored += 1
        for c in collars:
            tp, fp, fn = sr.standard_counts[c]
            std_counts[c][0] += tp
            std_counts[c][1] += fp
            std_counts[c][2] += fn
            mtp, mcount = sr.matched_counts[c]
            matched_counts[c][0] += mtp
            matched_counts[c][1] += mcount
        if sr.overall_mae_ms is not None:
            macro_mae.append(sr.overall_mae_ms)
        if sr.miou is not None:
            macro_miou.append(sr.miou)

        per_sample.append(
            {
                "key": sr.key,
                "num_pred_words": sr.num_pred_words,
                "num_gt_words": sr.num_gt_words,
                "num_aligned": sr.num_aligned,
                "overall_mae_ms": sr.overall_mae_ms,
                "median_ae_ms": sr.median_ae_ms,
                "miou": sr.miou,
                "median_iou": sr.median_iou,
            }
        )

    standard_f1 = {}
    matched_f1 = {}
    for c in collars:
        tp, fp, fn = std_counts[c]
        p, r, f1 = prf(tp, fp, fn)
        standard_f1[collar_label(c)] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        mtp, mcount = matched_counts[c]
        matched_f1[collar_label(c)] = {
            "f1": (mtp / mcount) if mcount > 0 else 0.0,
            "tp": mtp,
            "count": mcount,
        }

    word_err = [(a + b) / 2.0 for a, b in zip(all_start_err, all_end_err)]
    err_all = summarize(word_err, 1000.0)
    err_start = summarize(all_start_err, 1000.0)
    err_end = summarize(all_end_err, 1000.0)
    iou_stats = summarize(all_ious)
    n_scored_words = len(all_ious)

    overall = {
        "num_gt_samples": len(ground_truth),
        "num_samples_with_predictions": n_scored,
        "coverage": (n_scored / len(ground_truth)) if ground_truth else 0.0,
        "num_pred_words": n_pred_words,
        "num_gt_words": n_gt_words,
        "num_gt_excluded_events": n_gt_excluded,
        "num_aligned_words": n_aligned,
        "num_scored_words": n_scored_words,
        "word_recall": (n_scored_words / n_gt_words) if n_gt_words else 0.0,
        "mae_ms": {
            "overall": err_all["mean"],
            "start": err_start["mean"],
            "end": err_end["mean"],
            "start_bias": _mean_ms(all_start_signed),
            "end_bias": _mean_ms(all_end_signed),
        },
        "median_ae_ms": {
            "overall": err_all["median"],
            "start": err_start["median"],
            "end": err_end["median"],
        },
        "p90_ae_ms": {
            "overall": err_all["p90"],
            "start": err_start["p90"],
            "end": err_end["p90"],
        },
        "mae_ms_macro": float(np.mean(macro_mae)) if macro_mae else None,
        "miou": iou_stats["mean"],
        "median_iou": iou_stats["median"],
        "miou_macro": float(np.mean(macro_miou)) if macro_miou else None,
        "f1_standard": standard_f1,
        "f1_matched": matched_f1,
    }

    return BenchmarkResult(
        model=model,
        dataset=dataset,
        collars=collars,
        overall=overall,
        per_sample=per_sample,
        condition=dict(condition) if condition else {"noise": "clean"},
        metadata=dict(metadata) if metadata else {},
    )
