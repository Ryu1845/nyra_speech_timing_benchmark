"""Shared helpers for API provider timing generation.

Each ``generate_<provider>.py`` script defines a ``transcribe(audio_path)`` that
returns ``(word_timings, raw_response)`` where ``word_timings`` is a list of
``(word, start_s, end_s)`` tuples, and calls :func:`run_provider`.

For every dataset the runner writes two artifacts:

* ``predictions/<provider>_<dataset>.json`` — the canonical scoring format
  understood by ``nyra-timing score`` (``{model, dataset, predictions}``).
* ``generation/outputs/raw/<provider>_<dataset>.json`` — the raw provider
  responses plus the parsed word timings, for auditing/debugging.

Buckeye chunks are sliced from their parent recording so timings are
chunk-relative (matching the ground truth). API keys are read from the research
repo ``.env`` (override with ``NYRA_ENV_FILE``) or the process environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nyra_timing_benchmark.audio_io import slice_wav  # noqa: E402
from nyra_timing_benchmark.schema import SCHEMA_VERSION  # noqa: E402
from nyra_timing_benchmark.datasets import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    load_ground_truth,
)

DEFAULT_ENV_FILE = "/home/ubuntu/code/research/.env"
DEFAULT_DATASETS = ["timit", "buckeye"]

WordTiming = Tuple[str, float, float]
# transcribe(audio_path) -> (word_timings, raw_response)
TranscribeFn = Callable[[Path], Tuple[List[WordTiming], object]]


def load_env_file(path: Optional[str] = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file (without overriding existing)."""
    env_path = Path(os.environ.get("NYRA_ENV_FILE", path or DEFAULT_ENV_FILE))
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require_key(env_var: str, explicit: str = "") -> str:
    key = explicit or os.environ.get(env_var, "")
    if not key:
        raise SystemExit(
            f"Missing API key: set {env_var} in the environment or {DEFAULT_ENV_FILE}, "
            f"or pass --api-key."
        )
    return key


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset name (e.g. timit, buckeye) or 'all' for the default set.",
    )
    parser.add_argument("--split", default=None, help="Dataset split (default: test).")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples per dataset (0 = all).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT),
        help="Base directory for predictions/ and generation/outputs/raw/.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing outputs and start fresh.",
    )
    parser.add_argument("--api-key", default="", help="Override the provider API key.")


def datasets_from_arg(dataset: str) -> List[str]:
    return list(DEFAULT_DATASETS) if dataset == "all" else [dataset]


def _pred_path(base: Path, provider: str, dataset: str) -> Path:
    return base / "predictions" / f"{provider}_{dataset}.json"


def _raw_path(base: Path, provider: str, dataset: str) -> Path:
    return base / "generation" / "outputs" / "raw" / f"{provider}_{dataset}.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_predictions(
    path: Path, provider: str, dataset: str, predictions: Dict[str, list]
) -> None:
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "model": provider,
            "dataset": dataset,
            "condition": {"noise": "clean"},
            "metadata": {
                "system_type": "transcription",
                "uses_reference_text": False,
                "deployment": "hosted_api",
                "description": f"{provider} hosted STT API word timestamps",
            },
            "predictions": predictions,
        },
    )


def _load_partial(
    pred_path: Path, raw_path: Path
) -> Tuple[Dict[str, list], List[dict]]:
    predictions: Dict[str, list] = {}
    raw_rows: List[dict] = []
    if pred_path.exists():
        try:
            predictions = json.loads(pred_path.read_text()).get("predictions", {}) or {}
        except Exception:
            predictions = {}
    if raw_path.exists():
        try:
            raw_rows = json.loads(raw_path.read_text()) or []
        except Exception:
            raw_rows = []
    return predictions, raw_rows


def run_provider(
    provider_id: str, transcribe: TranscribeFn, args: argparse.Namespace
) -> None:
    """Run ``transcribe`` over every requested dataset and write both artifacts."""
    base = Path(args.out_dir)
    for dataset in datasets_from_arg(args.dataset):
        pred_path = _pred_path(base, provider_id, dataset)
        raw_path = _raw_path(base, provider_id, dataset)

        gt = load_ground_truth(dataset, args.data_root, split=args.split)
        keys = list(gt.keys())
        if args.max_samples:
            keys = keys[: args.max_samples]

        if args.no_resume:
            predictions, raw_rows = {}, []
        else:
            predictions, raw_rows = _load_partial(pred_path, raw_path)
        done = set(predictions.keys())

        print(
            f"\n[{provider_id} / {dataset}] {len(keys)} samples "
            f"({len(done)} already done)"
        )
        slice_dir: Optional[str] = None
        n_ok = 0
        try:
            for i, key in enumerate(keys, start=1):
                if key in done:
                    continue
                sample = gt[key]
                if not sample.audio_path or not os.path.exists(sample.audio_path):
                    print(f"  [{i}/{len(keys)}] {key}: audio missing, skipping")
                    continue

                audio_path = sample.audio_path
                if sample.chunk_start is not None and sample.chunk_end is not None:
                    if slice_dir is None:
                        slice_dir = tempfile.mkdtemp(prefix="nyra_gen_")
                    out_wav = os.path.join(slice_dir, f"{key.replace('/', '_')}.wav")
                    audio_path = slice_wav(
                        sample.audio_path,
                        sample.chunk_start,
                        sample.chunk_end,
                        out_wav,
                    )

                try:
                    triples, raw = transcribe(Path(audio_path))
                except Exception as e:  # keep going; record the failure
                    print(f"  [{i}/{len(keys)}] {key}: ERROR {type(e).__name__}: {e}")
                    triples, raw = [], {"error": f"{type(e).__name__}: {e}"}

                entries = [
                    {"word": w, "start": float(s), "end": float(e)}
                    for (w, s, e) in triples
                ]
                predictions[key] = entries
                raw_rows.append(
                    {
                        "id": key,
                        "audio": os.path.basename(sample.audio_path),
                        "dataset": dataset,
                        "split": args.split or "test",
                        "model": provider_id,
                        "words": entries,
                        "raw_response": raw,
                    }
                )
                if entries:
                    n_ok += 1
                if i == 1 or i % 10 == 0:
                    print(f"  [{i}/{len(keys)}] {key}: {len(entries)} words")
                if i % 25 == 0:
                    _write_predictions(pred_path, provider_id, dataset, predictions)
                    _write_json(raw_path, raw_rows)
        finally:
            if slice_dir:
                shutil.rmtree(slice_dir, ignore_errors=True)

        _write_predictions(pred_path, provider_id, dataset, predictions)
        _write_json(raw_path, raw_rows)
        print(
            f"  wrote {pred_path} ({len(predictions)} samples, "
            f"{n_ok} with words) and {raw_path}"
        )
