"""Ground-truth dataset registry and loading.

Ground truth is read at runtime from the existing data directory layout (no audio
is bundled into this repo). Two on-disk formats are supported:

* ``labels_json``: one audio file per sample with absolute (file-relative) word
  timings (TIMIT, FluencyBank).
* ``buckeye_chunks``: ``eval_chunks.json`` where each chunk is a sub-segment of a
  longer parent recording; word timings are *chunk-relative*, so the chunk must
  be sliced from the parent audio (see ``audio_io.slice_wav``) before inference.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .normalize import normalize_word

DEFAULT_DATA_ROOT = "/home/ubuntu/data/english_speech"


@dataclass(frozen=True)
class DatasetSpec:
    """Static configuration for a registered dataset."""

    name: str
    rel_path: str
    labels_file: str = "labels.json"
    default_split: str = "test"
    # Preference order of timing fields in labels.json.
    timing_fields: Tuple[str, ...] = ("verbatim_timings", "timings")
    # On-disk format: "labels_json" or "buckeye_chunks".
    data_format: str = "labels_json"


DATASETS: Dict[str, DatasetSpec] = {
    "timit": DatasetSpec(name="timit", rel_path="timit"),
    "fluency_bank_stutter": DatasetSpec(
        name="fluency_bank_stutter", rel_path="fluency_bank_stutter"
    ),
    "buckeye": DatasetSpec(
        name="buckeye",
        rel_path="buckeye",
        labels_file="eval_chunks.json",
        data_format="buckeye_chunks",
    ),
}


@dataclass
class GroundTruthSample:
    """Ground-truth words and timings for a single utterance.

    For chunked datasets (Buckeye), ``audio_path`` points at the parent recording
    and ``chunk_start``/``chunk_end`` (seconds) define the segment to slice; the
    word timings are relative to ``chunk_start``.
    """

    key: str
    words: List[str] = field(default_factory=list)
    timings: List[Tuple[float, float]] = field(default_factory=list)
    transcript: str = ""
    audio_path: Optional[str] = None
    chunk_start: Optional[float] = None
    chunk_end: Optional[float] = None
    #: Duration (s) of the audio the timings refer to (the chunk, if chunked).
    duration: Optional[float] = None


def list_datasets() -> List[str]:
    return sorted(DATASETS.keys())


def get_spec(dataset: str) -> DatasetSpec:
    if dataset not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(list_datasets())}"
        )
    return DATASETS[dataset]


def labels_path(dataset: str, data_root: str = DEFAULT_DATA_ROOT) -> str:
    spec = get_spec(dataset)
    return os.path.join(data_root, spec.rel_path, spec.labels_file)


def _buckeye_align_word(transcript_form: Optional[str]) -> str:
    """Map a Buckeye ``transcript_form`` to a plain spoken word.

    Mirrors the research baseline so timing numbers stay comparable:
    filled-pause/laughter markers become words and a trailing ``*`` (cut-off /
    mispronunciation marker) is stripped. Returns "" for empty input.
    """
    if not transcript_form:
        return ""
    tf = transcript_form.strip()
    mapping = {"[UM]": "um", "[UH]": "uh", "[laughter]": "laughter"}
    if tf in mapping:
        return mapping[tf]
    if tf.endswith("*"):
        tf = tf[:-1]
    return tf


def _extract_words_timings(
    timings: List[dict],
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """Convert labels.json timing entries to spoken word/timing lists.

    Drops spacer/empty entries (those whose normalized word is empty) and any
    entry missing start/end times.
    """
    words: List[str] = []
    times: List[Tuple[float, float]] = []
    for t in timings:
        raw_word = t.get("word")
        start = t.get("starttime")
        end = t.get("endtime")
        if start is None or end is None:
            continue
        if not normalize_word(raw_word):
            continue
        words.append(str(raw_word).strip())
        times.append((float(start), float(end)))
    return words, times


def _extract_buckeye_words_timings(
    timings: List[dict],
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """Convert Buckeye chunk timings using ``transcript_form`` normalization."""
    words: List[str] = []
    times: List[Tuple[float, float]] = []
    for t in timings:
        start = t.get("starttime")
        end = t.get("endtime")
        if start is None or end is None or end <= start:
            continue
        align_word = _buckeye_align_word(t.get("transcript_form") or t.get("word"))
        if not normalize_word(align_word):
            continue
        words.append(align_word.strip())
        times.append((float(start), float(end)))
    return words, times


def _load_labels_json(
    spec: DatasetSpec, labels: list, base_dir: str, split: Optional[str]
) -> "OrderedDict[str, GroundTruthSample]":
    samples: "OrderedDict[str, GroundTruthSample]" = OrderedDict()
    for sample in labels:
        if split is not None and sample.get("split") != split:
            continue
        timings = None
        for fieldname in spec.timing_fields:
            timings = sample.get(fieldname)
            if timings:
                break
        if not timings:
            continue
        words, times = _extract_words_timings(timings)
        if not words:
            continue
        key = sample.get("audio") or sample.get("id")
        if key is None:
            continue
        audio_rel = sample.get("audio")
        audio_path = os.path.join(base_dir, audio_rel) if audio_rel else None
        samples[key] = GroundTruthSample(
            key=key,
            words=words,
            timings=times,
            transcript=sample.get("transcript", "") or "",
            audio_path=audio_path,
            duration=sample.get("duration_in_s"),
        )
    return samples


def _load_buckeye_chunks(
    chunks: list, base_dir: str, split: Optional[str]
) -> "OrderedDict[str, GroundTruthSample]":
    samples: "OrderedDict[str, GroundTruthSample]" = OrderedDict()
    for chunk in chunks:
        if split is not None and chunk.get("split") != split:
            continue
        words, times = _extract_buckeye_words_timings(chunk.get("timings", []))
        if not words:
            continue
        key = chunk.get("id")
        if key is None:
            continue
        audio_rel = chunk.get("audio")
        audio_path = os.path.join(base_dir, audio_rel) if audio_rel else None
        samples[key] = GroundTruthSample(
            key=key,
            words=words,
            timings=times,
            transcript=chunk.get("transcript", "") or "",
            audio_path=audio_path,
            chunk_start=chunk.get("chunk_start_s"),
            chunk_end=chunk.get("chunk_end_s"),
            duration=chunk.get("duration_s"),
        )
    return samples


def load_ground_truth(
    dataset: str,
    data_root: str = DEFAULT_DATA_ROOT,
    split: Optional[str] = None,
) -> "Dict[str, GroundTruthSample]":
    """Load ground-truth samples keyed by their sample id.

    Only samples in ``split`` (default: the dataset's default split) that have at
    least one spoken word with timings are returned.
    """
    spec = get_spec(dataset)
    split = split or spec.default_split
    path = labels_path(dataset, data_root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labels file not found for dataset '{dataset}': {path}. "
            f"Pass a correct --data-root."
        )

    with open(path, "r") as f:
        labels = json.load(f)

    base_dir = os.path.join(data_root, spec.rel_path)
    if spec.data_format == "buckeye_chunks":
        return _load_buckeye_chunks(labels, base_dir, split)
    return _load_labels_json(spec, labels, base_dir, split)
