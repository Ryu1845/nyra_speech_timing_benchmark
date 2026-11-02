"""Public, text-free sample manifests.

The corpora (TIMIT, Buckeye, FluencyBank) are licensed, so neither audio nor
transcripts/timings can be published. A manifest lists only what an external
submitter needs to produce predictions from *their own* licensed copy and to
self-validate the result:

* ``key``           -- the sample key predictions must use,
* ``audio``         -- audio path relative to the dataset dir in this benchmark's
                       data layout,
* ``corpus_path``   -- path in the original distribution, where it is derivable
                       (TIMIT: ``TEST/DR4/MGMM0/SX139.WAV``),
* ``chunk_start_s`` / ``chunk_end_s`` -- segment of ``audio`` the sample covers
                       (Buckeye); predicted times are relative to the chunk start,
* ``duration_s``    -- duration of the (chunked) sample audio.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

from .datasets import DEFAULT_DATA_ROOT, get_spec, load_ground_truth

MANIFEST_VERSION = "1.0"

_TIMIT_KEY = re.compile(r"(TRAIN|TEST)_(DR\d)_([A-Z0-9]+)_([A-Z0-9]+)\.wav$", re.I)


def _corpus_path(dataset: str, audio_rel: str) -> Optional[str]:
    if dataset == "timit":
        m = _TIMIT_KEY.search(audio_rel)
        if m:
            return "/".join(g.upper() for g in m.groups()) + ".WAV"
    if dataset == "buckeye":
        return audio_rel
    return None


def build_manifest(
    dataset: str, data_root: str = DEFAULT_DATA_ROOT, split: Optional[str] = None
) -> Dict:
    spec = get_spec(dataset)
    split = split or spec.default_split
    gt = load_ground_truth(dataset, data_root, split=split)
    base_dir = os.path.join(data_root, spec.rel_path)
    samples = []
    for key, s in gt.items():
        audio_rel = os.path.relpath(s.audio_path, base_dir) if s.audio_path else None
        entry = {
            "key": key,
            "audio": audio_rel,
            "corpus_path": _corpus_path(dataset, audio_rel or ""),
            "duration_s": s.duration,
        }
        if s.chunk_start is not None:
            entry["chunk_start_s"] = s.chunk_start
            entry["chunk_end_s"] = s.chunk_end
        samples.append(entry)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "split": split,
        "num_samples": len(samples),
        "samples": samples,
    }


def write_manifest(manifest: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    return path


def load_manifest(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def manifest_durations(manifest: Dict) -> Dict[str, Optional[float]]:
    return {s["key"]: s.get("duration_s") for s in manifest["samples"]}
