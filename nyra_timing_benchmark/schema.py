"""Prediction / submission file schema, loading and validation.

A predictions file is JSON with the shape::

    {
      "schema_version": "1.0",
      "model": "my_model",
      "dataset": "timit",
      "condition": {"noise": "clean"},
      "metadata": {
        "system_type": "forced_aligner",
        "uses_reference_text": true,
        "description": "...",
        "organization": "...",
        "contact": "...",
        "model_url": "...",
        "paper_url": "...",
        "license": "...",
        "open_weights": true
      },
      "predictions": {
        "<sample_key>": [
          {"word": "the", "start": 0.14, "end": 0.20},
          ...
        ]
      }
    }

Only ``model`` and ``predictions`` are required to *score* a file (legacy files
without ``schema_version``/``metadata`` still load). A leaderboard *submission*
additionally needs ``dataset`` and ``metadata.system_type``; use
:func:`validate_submission` (``nyra-timing validate``) to check one.

``<sample_key>`` must match the ground-truth sample key (the ``key`` field of the
dataset manifest). Each word entry needs ``word``, ``start`` and ``end`` in
seconds, relative to the start of the (possibly chunked) sample audio.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

SCHEMA_VERSION = "1.0"
SYSTEM_TYPES = ("forced_aligner", "transcription")
# Predictions may run slightly past the audio end (frame rounding, padding).
DURATION_TOLERANCE_S = 0.5
# Models do emit the odd word after the audio ends; only a systematic overrun
# (ms instead of s, absolute instead of chunk-relative times) is a format error.
LATE_START_ERROR_FRACTION = 0.2


@dataclass
class PredictionFile:
    """Parsed predictions for a single model/dataset/condition."""

    model: str
    dataset: Optional[str]
    # sample_key -> (words, [(start, end), ...])
    predictions: Dict[str, Tuple[List[str], List[Tuple[float, float]]]] = field(
        default_factory=dict
    )
    condition: Dict = field(default_factory=lambda: {"noise": "clean"})
    metadata: Dict = field(default_factory=dict)
    schema_version: Optional[str] = None


@dataclass
class ValidationReport:
    """Result of validating a predictions file against ground truth."""

    num_pred_samples: int
    num_gt_samples: int
    matched_keys: List[str]
    unknown_keys: List[str]
    missing_keys: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.num_gt_samples == 0:
            return 0.0
        return len(self.matched_keys) / self.num_gt_samples

    @property
    def ok(self) -> bool:
        return not self.errors


def read_json(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _parse_word_entries(entries) -> Tuple[List[str], List[Tuple[float, float]]]:
    words: List[str] = []
    times: List[Tuple[float, float]] = []
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError(f"Each prediction entry must be an object, got: {e!r}")
        if "word" not in e or "start" not in e or "end" not in e:
            raise ValueError(
                f"Prediction entry missing required keys (word/start/end): {e!r}"
            )
        words.append(str(e["word"]))
        times.append((float(e["start"]), float(e["end"])))
    return words, times


def parse_predictions(data: Mapping) -> PredictionFile:
    """Parse an already-loaded predictions JSON object."""
    if "predictions" not in data or not isinstance(data["predictions"], dict):
        raise ValueError(
            "Predictions file must contain a 'predictions' object mapping "
            "sample keys to lists of word entries."
        )

    parsed: Dict[str, Tuple[List[str], List[Tuple[float, float]]]] = {}
    for key, entries in data["predictions"].items():
        if not isinstance(entries, list):
            raise ValueError(f"Predictions for '{key}' must be a list.")
        parsed[key] = _parse_word_entries(entries)

    condition = data.get("condition") or {"noise": "clean"}
    return PredictionFile(
        model=str(data.get("model", "model")),
        dataset=data.get("dataset"),
        predictions=parsed,
        condition=dict(condition),
        metadata=dict(data.get("metadata") or {}),
        schema_version=data.get("schema_version"),
    )


def load_predictions(path: str) -> PredictionFile:
    """Load and parse a predictions JSON file (``.json`` or ``.json.gz``)."""
    return parse_predictions(read_json(path))


def write_predictions(
    path: str,
    predictions: Mapping[str, Tuple[List[str], List[Tuple[float, float]]]],
    model: str,
    dataset: str,
    condition: Optional[Mapping] = None,
    metadata: Optional[Mapping] = None,
) -> str:
    """Write predictions in the canonical schema."""
    entries: Dict[str, list] = {}
    for key, (words, times) in predictions.items():
        entries[key] = [
            {"word": w, "start": float(s), "end": float(e)}
            for w, (s, e) in zip(words, times)
        ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "dataset": dataset,
        "condition": dict(condition) if condition else {"noise": "clean"},
        "metadata": dict(metadata) if metadata else {},
        "predictions": entries,
    }
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=None if path.endswith(".gz") else 2)
    return path


def validate_predictions(
    pred: PredictionFile, gt_keys: List[str]
) -> ValidationReport:
    """Compare prediction keys against ground-truth keys."""
    gt_set = set(gt_keys)
    pred_set = set(pred.predictions.keys())
    matched = sorted(pred_set & gt_set)
    unknown = sorted(pred_set - gt_set)
    missing = sorted(gt_set - pred_set)
    return ValidationReport(
        num_pred_samples=len(pred_set),
        num_gt_samples=len(gt_set),
        matched_keys=matched,
        unknown_keys=unknown,
        missing_keys=missing,
    )


def _preview(items: List[str], n: int = 3) -> str:
    more = f" (+{len(items) - n} more)" if len(items) > n else ""
    return ", ".join(items[:n]) + more


def validate_submission(
    data: Mapping,
    durations: Mapping[str, Optional[float]],
    dataset: Optional[str] = None,
) -> ValidationReport:
    """Strictly validate a leaderboard submission.

    ``durations`` maps every expected sample key to its audio duration in
    seconds (or ``None`` if unknown); it can come from the private ground truth
    or from the public manifest, so submitters can self-check without the
    labels. Errors make the file unscorable; warnings are reported but allowed.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if data.get("schema_version") not in (None, SCHEMA_VERSION):
        errors.append(
            f"Unsupported schema_version {data.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION!r})."
        )
    elif data.get("schema_version") is None:
        warnings.append(f"No schema_version; assuming {SCHEMA_VERSION}.")
    if not data.get("model"):
        errors.append("Missing 'model'.")
    if not data.get("dataset"):
        errors.append("Missing 'dataset'.")
    elif dataset and data["dataset"] != dataset:
        errors.append(f"dataset {data['dataset']!r} != expected {dataset!r}.")

    meta = data.get("metadata") or {}
    st = meta.get("system_type")
    if st not in SYSTEM_TYPES:
        errors.append(
            f"metadata.system_type must be one of {SYSTEM_TYPES}, got {st!r}."
        )
    if "uses_reference_text" not in meta:
        warnings.append(
            "metadata.uses_reference_text not set; inferred from system_type."
        )
    for k in ("description", "model_url", "license"):
        if not meta.get(k):
            warnings.append(f"metadata.{k} is empty.")

    cond = data.get("condition") or {"noise": "clean"}
    if cond.get("noise", "clean") != "clean" and cond.get("snr_db") is None:
        errors.append("Noisy condition requires condition.snr_db.")

    preds = data.get("predictions")
    if not isinstance(preds, dict):
        errors.append("Missing 'predictions' object.")
        preds = {}

    bad_entries: List[str] = []
    bad_times: List[str] = []
    unordered: List[str] = []
    starts_late: List[str] = []
    ends_late: List[str] = []
    empty_words = 0
    for key, entries in preds.items():
        if not isinstance(entries, list):
            bad_entries.append(key)
            continue
        prev_start = -math.inf
        for e in entries:
            if not isinstance(e, dict) or not {"word", "start", "end"} <= set(e):
                bad_entries.append(key)
                break
            try:
                s, en = float(e["start"]), float(e["end"])
            except (TypeError, ValueError):
                bad_times.append(key)
                break
            if not (math.isfinite(s) and math.isfinite(en)) or s < 0 or en < s:
                bad_times.append(key)
                break
            if not str(e["word"]).strip():
                empty_words += 1
            if s < prev_start - 1e-6:
                unordered.append(key)
            prev_start = s
            dur = durations.get(key)
            if dur is not None and s > dur + DURATION_TOLERANCE_S:
                starts_late.append(key)
                break
            if dur is not None and en > dur + DURATION_TOLERANCE_S:
                ends_late.append(key)

    if bad_entries:
        errors.append(
            f"{len(bad_entries)} samples have malformed word entries "
            f"(need word/start/end): {_preview(bad_entries)}"
        )
    if bad_times:
        errors.append(
            f"{len(bad_times)} samples have invalid times (non-finite, negative "
            f"or end < start): {_preview(bad_times)}"
        )
    if starts_late:
        msg = (
            f"{len(starts_late)} samples have words starting more than "
            f"{DURATION_TOLERANCE_S}s past the audio end: {_preview(starts_late)}"
        )
        if len(starts_late) > LATE_START_ERROR_FRACTION * max(1, len(preds)):
            errors.append(
                msg + " -- systematic; wrong keys, absolute instead of "
                "chunk-relative times, or ms instead of s?"
            )
        else:
            warnings.append(msg + " (scored as-is)")
    ends_late = sorted(set(ends_late) - set(starts_late))
    if ends_late:
        warnings.append(
            f"{len(ends_late)} samples have a word ending more than "
            f"{DURATION_TOLERANCE_S}s past the audio end (scored as-is): "
            f"{_preview(ends_late)}"
        )
    if unordered:
        warnings.append(
            f"{len(set(unordered))} samples have words not sorted by start time: "
            f"{_preview(sorted(set(unordered)))}"
        )
    if empty_words:
        warnings.append(f"{empty_words} word entries are empty strings (ignored).")

    expected = set(durations)
    got = set(preds)
    unknown = sorted(got - expected)
    missing = sorted(expected - got)
    if unknown:
        errors.append(
            f"{len(unknown)} sample keys are not in the dataset: {_preview(unknown)}"
        )
    if missing:
        warnings.append(
            f"{len(missing)}/{len(expected)} samples have no prediction and will "
            f"be scored as empty: {_preview(missing)}"
        )

    return ValidationReport(
        num_pred_samples=len(got),
        num_gt_samples=len(expected),
        matched_keys=sorted(got & expected),
        unknown_keys=unknown,
        missing_keys=missing,
        errors=errors,
        warnings=warnings,
    )
