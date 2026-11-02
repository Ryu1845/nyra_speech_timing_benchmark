"""Command-line interface for the timing benchmark.

Subcommands:
    list-datasets    show registered datasets and their test-sample counts
    export-manifest  write the public, text-free sample manifest for a dataset
    validate         check a submission file against GT or a public manifest
    score            score one or more predictions files against a dataset
    run              run a built-in model adapter (optionally under noise), score
    robustness       run a model over a noise x SNR grid and tabulate degradation
    make-noisy       render noisy copies of a dataset's audio for external models
    compare          build a leaderboard from several result JSON files
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .audio_io import slice_wav
from .datasets import (
    DEFAULT_DATA_ROOT,
    get_spec,
    labels_path,
    list_datasets,
    load_ground_truth,
)
from .evaluate import DEFAULT_COLLARS, BenchmarkResult, evaluate_dataset
from .noise import CLEAN, NoiseBank, NoiseCondition, parse_snrs, render_sample
from .schema import (
    read_json,
    load_predictions,
    validate_predictions,
    validate_submission,
    write_predictions,
)


def _parse_collars(value: Optional[str]) -> List[float]:
    if not value:
        return list(DEFAULT_COLLARS)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _fmt(v, suffix: str = "", pct: bool = False, nd: int = 1) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.{nd}f}%"
    return f"{v:.{nd}f}{suffix}"


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _result_stem(model: str, dataset: str, condition: Optional[Dict]) -> str:
    label = NoiseCondition.from_dict(condition).label
    suffix = "" if label == CLEAN else f"_{label}"
    return f"{_safe(model)}_{dataset}{suffix}"


def _print_summary(result: BenchmarkResult) -> None:
    o = result.overall
    mae = o["mae_ms"]
    med = o.get("median_ae_ms") or {}
    p90 = o.get("p90_ae_ms") or {}
    print(f"\n{'=' * 64}")
    print(f"Model:     {result.model}")
    print(f"Dataset:   {result.dataset}")
    print(f"Condition: {NoiseCondition.from_dict(result.condition).label}")
    print(f"{'=' * 64}")
    print(
        f"Samples: {o['num_samples_with_predictions']}/{o['num_gt_samples']} "
        f"with predictions (coverage {_fmt(o['coverage'], pct=True)})"
    )
    print(
        f"Words:   pred={o['num_pred_words']}  gt={o['num_gt_words']}  "
        f"scored={o['num_scored_words']}  "
        f"(word recall {_fmt(o['word_recall'], pct=True)})"
    )
    print("\nPer-word timing error ((|d_start|+|d_end|)/2):")
    print(f"  {'':8} {'mean':>8} {'median':>8} {'p90':>8}")
    for k in ("overall", "start", "end"):
        print(
            f"  {k:<8} {_fmt(mae[k], 'ms'):>8} {_fmt(med.get(k), 'ms'):>8} "
            f"{_fmt(p90.get(k), 'ms'):>8}"
        )
    print(f"  bias     start {_fmt(mae['start_bias'], 'ms')}  end {_fmt(mae['end_bias'], 'ms')}")
    print(f"  macro    {_fmt(o['mae_ms_macro'], 'ms')} (mean per-sample MAE)")
    print(
        f"IoU:     mean {_fmt(o['miou'], pct=True)}  "
        f"median {_fmt(o.get('median_iou'), pct=True)}"
    )

    print("\nF1 @ collar (standard, transcription-coupled):")
    for label, m in o["f1_standard"].items():
        print(
            f"  F1@{label:<6} {_fmt(m['f1'], pct=True):>7}  "
            f"(P {_fmt(m['precision'], pct=True)} / R {_fmt(m['recall'], pct=True)})"
        )

    print("\nCollar accuracy (matched words, timing-only):")
    for label, m in o["f1_matched"].items():
        print(f"  Acc@{label:<5} {_fmt(m['f1'], pct=True):>7}  (tp {m['tp']}/{m['count']})")
    print()


def _save_result(result: BenchmarkResult, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = _result_stem(result.model, result.dataset or "dataset", result.condition)
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def _save_predictions(
    predictions: Dict[str, tuple],
    model: str,
    dataset: str,
    out_dir: str,
    condition: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> str:
    """Write predictions in the canonical schema (see ``schema.py``) so a run is
    independently re-scorable via ``nyra-timing score --predictions ...``."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_result_stem(model, dataset, condition)}.json")
    return write_predictions(path, predictions, model, dataset, condition, metadata)


def _add_noise_args(p: argparse.ArgumentParser, multi: bool = False) -> None:
    if multi:
        p.add_argument(
            "--noise",
            default="gaussian",
            help="Comma-separated noise types: 'gaussian' and/or category names "
            "under --noise-dir.",
        )
        p.add_argument(
            "--snrs",
            default="clean,30,20,10,5,0",
            help="Comma-separated SNRs in dB; 'clean' adds the noise-free run.",
        )
    else:
        p.add_argument(
            "--noise",
            default=CLEAN,
            help="'clean' (default), 'gaussian', or a category under --noise-dir.",
        )
        p.add_argument("--snr", type=float, default=None, help="SNR in dB.")
    p.add_argument(
        "--noise-dir",
        default=None,
        help="Directory with one sub-directory of noise files per category.",
    )
    p.add_argument("--noise-seed", type=int, default=0)


def cmd_list_datasets(args: argparse.Namespace) -> int:
    print(f"Data root: {args.data_root}\n")
    print(f"{'dataset':<24} {'split':<8} {'test_samples':<14} labels.json")
    print("-" * 72)
    for name in list_datasets():
        spec = get_spec(name)
        path = labels_path(name, args.data_root)
        if os.path.exists(path):
            try:
                gt = load_ground_truth(name, args.data_root)
                count = str(len(gt))
            except Exception as e:  # pragma: no cover
                count = f"error: {e}"
        else:
            count = "missing"
        print(f"{name:<24} {spec.default_split:<8} {count:<14} {path}")
    return 0


def cmd_export_manifest(args: argparse.Namespace) -> int:
    from .manifest import build_manifest, write_manifest

    names = list_datasets() if args.dataset == "all" else [args.dataset]
    for name in names:
        m = build_manifest(name, args.data_root, split=args.split)
        path = write_manifest(m, os.path.join(args.out, f"{name}.json"))
        print(f"{name}: {m['num_samples']} samples -> {path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    data = read_json(args.predictions)
    dataset = args.dataset or data.get("dataset")
    if args.manifest:
        from .manifest import load_manifest, manifest_durations

        manifest = load_manifest(args.manifest)
        dataset = dataset or manifest.get("dataset")
        durations = manifest_durations(manifest)
        source = f"manifest {args.manifest}"
    else:
        if not dataset:
            print("Pass --dataset or --manifest.", file=sys.stderr)
            return 2
        gt = load_ground_truth(dataset, args.data_root, split=args.split)
        durations = {k: s.duration for k, s in gt.items()}
        source = f"ground truth ({dataset})"

    report = validate_submission(data, durations, dataset=dataset)
    print(f"Validated {args.predictions} against {source}")
    print(
        f"  {len(report.matched_keys)}/{report.num_gt_samples} samples covered, "
        f"{len(report.unknown_keys)} unknown keys"
    )
    for w in report.warnings:
        print(f"  WARNING: {w}")
    for e in report.errors:
        print(f"  ERROR:   {e}")
    print("  OK" if report.ok else "  FAILED")
    return 0 if report.ok else 1


def cmd_score(args: argparse.Namespace) -> int:
    collars = _parse_collars(args.collars)
    gt_cache: Dict[str, dict] = {}
    status = 0
    for ppath in args.predictions:
        pred = load_predictions(ppath)
        dataset = args.dataset or pred.dataset
        if not dataset:
            print(f"{ppath}: no dataset in file; pass --dataset.", file=sys.stderr)
            status = 2
            continue
        if pred.dataset and pred.dataset != dataset:
            print(
                f"Warning: predictions file dataset '{pred.dataset}' != "
                f"--dataset '{dataset}'.",
                file=sys.stderr,
            )
        if dataset not in gt_cache:
            gt_full = load_ground_truth(dataset, args.data_root, split=args.split)
            if args.subset:
                gt_full = {k: gt_full[k] for k in _subset_keys(gt_full, args.subset)}
            gt_cache[dataset] = gt_full
        gt = gt_cache[dataset]
        if args.subset:
            pred.predictions = {k: v for k, v in pred.predictions.items() if k in gt}

        model_name = args.model_name or pred.model
        report = validate_predictions(pred, list(gt.keys()))
        print(
            f"Validation: {len(report.matched_keys)} matched, "
            f"{len(report.unknown_keys)} unknown keys, "
            f"{len(report.missing_keys)} GT samples missing predictions."
        )
        if report.unknown_keys:
            preview = ", ".join(report.unknown_keys[:3])
            print(f"  e.g. unknown keys: {preview} ...", file=sys.stderr)

        result = evaluate_dataset(
            gt,
            pred.predictions,
            model=model_name,
            dataset=dataset,
            collars=collars,
            condition=pred.condition,
            metadata=pred.metadata,
        )
        _print_summary(result)
        if args.out:
            path = _save_result(result, args.out)
            print(f"Saved results to: {path}")
    return status


def _model_kwargs(args: argparse.Namespace) -> Dict:
    if args.model != "crisperwhisper":
        return {}
    kw = {
        "model_path": args.crisperwhisper_model,
        "repo_path": args.crisperwhisper_repo,
        "backend": args.crisperwhisper_backend,
        "cache_dir": args.crisperwhisper_cache,
        "device": args.device,
        "compute_type": args.compute_type,
    }
    return {k: v for k, v in kw.items() if v is not None}


def _builtin_metadata(model_name: str) -> Dict:
    from .models.registry import HOSTED_MODELS, MODEL_IS_FORCED_ALIGNER

    fa = MODEL_IS_FORCED_ALIGNER.get(model_name, False)
    return {
        "system_type": "forced_aligner" if fa else "transcription",
        "uses_reference_text": fa,
        "deployment": "hosted_api" if model_name in HOSTED_MODELS else "local",
        "description": f"built-in '{model_name}' adapter",
    }


def _build_items(
    gt: Dict,
    keys: Sequence[str],
    work_dir: str,
    condition: NoiseCondition,
    bank: Optional[NoiseBank],
):
    """Materialise the audio each model sees under ``condition``.

    Clean, unchunked samples are passed through untouched so clean numbers stay
    byte-identical to earlier runs; everything else is rendered to ``work_dir``.
    """
    from .models.base import PredictItem

    items = []
    for key in keys:
        sample = gt[key]
        if not sample.audio_path or not os.path.exists(sample.audio_path):
            print(f"  audio not found, skipping {key}: {sample.audio_path}")
            continue
        chunk = None
        if sample.chunk_start is not None and sample.chunk_end is not None:
            chunk = (sample.chunk_start, sample.chunk_end)
        audio_path = sample.audio_path
        out_wav = os.path.join(work_dir, f"{_safe(key)}.wav")
        try:
            if not condition.is_clean:
                audio_path = render_sample(
                    sample.audio_path, out_wav, key, condition, bank, chunk
                )
            elif chunk is not None:
                audio_path = slice_wav(sample.audio_path, chunk[0], chunk[1], out_wav)
        except Exception as e:
            print(f"  audio prep failed for {key}: {e}")
            continue
        items.append(
            PredictItem(
                key=key,
                audio_path=audio_path,
                words=list(sample.words),
                transcript=sample.transcript,
            )
        )
    return items


def _load_resume(model_name, dataset, condition, pred_dir):
    """Existing predictions and raw responses saved by an earlier, partial run."""
    stem = _result_stem(model_name, dataset, condition.to_dict())
    path = os.path.join(pred_dir, f"{stem}.json")
    if not os.path.exists(path):
        return {}, {}
    prior = load_predictions(path).predictions
    raw_path = os.path.join(pred_dir, "raw", f"{stem}.json.gz")
    raw = read_json(raw_path) if os.path.exists(raw_path) else {}
    return prior, {k: v for k, v in raw.items() if k in prior}


def _predict_condition(
    model, gt, keys, condition, bank, tmp_root, resume_dir=None, dataset=None
) -> Dict[str, tuple]:
    prior, prior_raw = {}, {}
    if resume_dir:
        prior, prior_raw = _load_resume(model.name, dataset, condition, resume_dir)
        keys = [k for k in keys if k not in prior]
        print(f"Resuming: {len(prior)} cached, {len(keys)} to predict [{condition.label}]")
    work_dir = tempfile.mkdtemp(prefix=f"nyra_{condition.label}_", dir=tmp_root)
    try:
        items = _build_items(gt, keys, work_dir, condition, bank)
        print(f"Running {model.name} on {len(items)} utterances [{condition.label}]...")
        raw = model.predict_many(items) if items else {}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    if prior_raw and getattr(model, "raw_responses", None) is not None:
        model.raw_responses = {**prior_raw, **model.raw_responses}
    new = {key: ([w for w, _, _ in t], [(s, e) for _, s, e in t]) for key, t in raw.items()}
    return {**prior, **new}


def _finish_condition(args, model, gt_subset, predictions, condition, metadata, collars):
    result = evaluate_dataset(
        gt_subset,
        predictions,
        model=model.name,
        dataset=args.dataset,
        collars=collars,
        condition=condition.to_dict(),
        metadata=metadata,
    )
    _print_summary(result)
    if args.out:
        print(f"Saved results to: {_save_result(result, args.out)}")
    if getattr(args, "save_predictions", None):
        ppath = _save_predictions(
            predictions,
            model.name,
            args.dataset,
            args.save_predictions,
            condition.to_dict(),
            metadata,
        )
        print(f"Saved predictions to: {ppath}")
        raw = getattr(model, "raw_responses", None)
        if raw:
            rpath = os.path.join(
                args.save_predictions,
                "raw",
                f"{_result_stem(model.name, args.dataset, condition.to_dict())}.json.gz",
            )
            os.makedirs(os.path.dirname(rpath), exist_ok=True)
            with gzip.open(rpath, "wt", encoding="utf-8") as f:
                json.dump(raw, f)
            print(f"Saved raw responses to: {rpath}")
    return result


def _subset_keys(keys: Sequence[str], n: Optional[int]) -> List[str]:
    """``n`` evenly spaced keys from the sorted key list (deterministic and,
    for TIMIT, spread over all dialect regions, unlike the first ``n``)."""
    keys = sorted(keys)
    if not n or n >= len(keys):
        return keys
    step = len(keys) / n
    return [keys[int(i * step)] for i in range(n)]


def _load_run_inputs(args):
    gt = load_ground_truth(args.dataset, args.data_root, split=args.split)
    keys = list(gt.keys())
    if args.subset:
        keys = _subset_keys(keys, args.subset)
    if args.num_samples:
        keys = keys[: args.num_samples]
    bank = NoiseBank(args.noise_dir) if args.noise_dir else None
    return gt, keys, {k: gt[k] for k in keys}, bank


def _resume_dir(args) -> Optional[str]:
    if not getattr(args, "resume", False):
        return None
    if not args.save_predictions:
        raise SystemExit("--resume needs --save-predictions (the directory to resume from)")
    return args.save_predictions


def cmd_run(args: argparse.Namespace) -> int:
    from .models.registry import build_model

    collars = _parse_collars(args.collars)
    condition = NoiseCondition(args.noise, args.snr, args.noise_seed)
    gt, keys, gt_subset, bank = _load_run_inputs(args)
    model = build_model(args.model, **_model_kwargs(args))
    try:
        predictions = _predict_condition(
            model, gt, keys, condition, bank, args.tmp_dir, _resume_dir(args), args.dataset
        )
    finally:
        model.close()
    _finish_condition(
        args, model, gt_subset, predictions, condition, _builtin_metadata(args.model), collars
    )
    return 0


def _robustness_row(result: BenchmarkResult) -> Dict:
    o = result.overall
    cond = NoiseCondition.from_dict(result.condition)
    return {
        "condition": cond.label,
        "noise": cond.noise,
        "snr_db": cond.snr_db,
        "mae_ms": o["mae_ms"]["overall"],
        "median_ae_ms": o["median_ae_ms"]["overall"],
        "p90_ae_ms": o["p90_ae_ms"]["overall"],
        "miou": o["miou"],
        "median_iou": o["median_iou"],
        "word_recall": o["word_recall"],
        "f1_standard": {k: v["f1"] for k, v in o["f1_standard"].items()},
    }


def _print_robustness(rows: List[Dict], model: str, dataset: str) -> None:
    labels = list(rows[0]["f1_standard"].keys())[:3] if rows else []
    print(f"\nRobustness: {model} on {dataset}")
    head = (
        f"{'condition':<22} {'MAE':>7} {'MedAE':>7} {'P90':>7} {'mIoU':>7} "
        f"{'MedIoU':>7} {'WordRec':>8}"
    )
    head += "".join(f" {'F1@' + lbl:>9}" for lbl in labels)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (
            f"{r['condition']:<22} {_fmt(r['mae_ms']):>7} {_fmt(r['median_ae_ms']):>7} "
            f"{_fmt(r['p90_ae_ms']):>7} {_fmt(r['miou'], pct=True):>7} "
            f"{_fmt(r['median_iou'], pct=True):>7} {_fmt(r['word_recall'], pct=True):>8}"
        )
        line += "".join(f" {_fmt(r['f1_standard'][lbl], pct=True):>9}" for lbl in labels)
        print(line)
    print()


def cmd_robustness(args: argparse.Namespace) -> int:
    from .models.registry import build_model

    collars = _parse_collars(args.collars)
    noises = [n.strip() for n in args.noise.split(",") if n.strip()]
    snrs = parse_snrs(args.snrs)
    conditions: List[NoiseCondition] = []
    if None in snrs:
        conditions.append(NoiseCondition())
    for n in noises:
        conditions += [NoiseCondition(n, s, args.noise_seed) for s in snrs if s is not None]

    gt, keys, gt_subset, bank = _load_run_inputs(args)
    metadata = _builtin_metadata(args.model)
    model = build_model(args.model, **_model_kwargs(args))
    rows: List[Dict] = []
    try:
        for cond in conditions:
            preds = _predict_condition(
                model, gt, keys, cond, bank, args.tmp_dir, _resume_dir(args), args.dataset
            )
            result = _finish_condition(args, model, gt_subset, preds, cond, metadata, collars)
            rows.append(_robustness_row(result))
    finally:
        model.close()

    _print_robustness(rows, model.name, args.dataset)
    if args.out:
        path = os.path.join(
            args.out, f"robustness_{_safe(model.name)}_{args.dataset}.json"
        )
        with open(path, "w") as f:
            json.dump(
                {"model": model.name, "dataset": args.dataset, "rows": rows}, f, indent=2
            )
        print(f"Saved robustness summary to: {path}")
    return 0


def _audio_seconds(path: str) -> float:
    import soundfile as sf

    try:
        return float(sf.info(path).duration)
    except Exception:
        import librosa

        return float(librosa.get_duration(path=path))


def _predict_one(model, item):
    if model.requires_transcript:
        return model.predict(item.audio_path, transcript=item.transcript, words=item.words)
    return model.predict(item.audio_path)


def _sample_seconds(sample) -> float:
    if sample.chunk_start is not None and sample.chunk_end is not None:
        return float(sample.chunk_end - sample.chunk_start)
    return _audio_seconds(sample.audio_path)


def cmd_speed(args: argparse.Namespace) -> int:
    """Real-time factor: processing seconds per second of audio, one clip at a
    time (no batching or concurrency). Model load and warm-up are excluded.
    Runs on the ``--longest`` N clips of the split."""
    import platform
    import time

    from .models.registry import HOSTED_MODELS, build_model

    gt = load_ground_truth(args.dataset, args.data_root, split=args.split)
    usable = [k for k in sorted(gt) if gt[k].audio_path and os.path.exists(gt[k].audio_path)]
    keys = sorted(usable, key=lambda k: -_sample_seconds(gt[k]))[: args.longest]
    model = build_model(args.model, **_model_kwargs(args))
    work_dir = tempfile.mkdtemp(prefix="nyra_speed_", dir=args.tmp_dir)
    try:
        items = _build_items(gt, keys, work_dir, NoiseCondition(), None)
        durations = {it.key: _audio_seconds(it.audio_path) for it in items}
        warm = sorted(usable, key=lambda k: _sample_seconds(gt[k]))[: args.warmup]
        for it in _build_items(gt, warm, work_dir, NoiseCondition(), None):
            _predict_one(model, it)

        latencies = []
        for n, it in enumerate(items, start=1):
            t0 = time.perf_counter()
            try:
                _predict_one(model, it)
            except Exception as e:
                print(f"  error on {it.key}: {type(e).__name__}: {e}")
                continue
            latencies.append((time.perf_counter() - t0, durations[it.key]))
            if n % 25 == 0:
                print(f"  [{n}/{len(items)}]", flush=True)
        single_rtf = sum(t for t, _ in latencies) / sum(d for _, d in latencies)
    finally:
        model.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    lat = sorted(t for t, _ in latencies)
    out = {
        "model": model.name,
        "dataset": args.dataset,
        "num_clips": len(latencies),
        "num_errors": len(items) - len(latencies),
        "audio_seconds": round(sum(d for _, d in latencies), 2),
        "mean_clip_seconds": round(sum(d for _, d in latencies) / max(1, len(latencies)), 2),
        "rtf": single_rtf,
        "median_latency_s": lat[len(lat) // 2] if lat else None,
        "deployment": "hosted_api" if args.model in HOSTED_MODELS else "local",
        "device": gpu or platform.processor() or "cpu",
        "date": time.strftime("%Y-%m-%d"),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"speed_{_safe(model.name)}_{args.dataset}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved speed results to: {path}")
    return 0


def cmd_make_noisy(args: argparse.Namespace) -> int:
    """Render noisy audio so models outside this repo can be run on it."""
    from .manifest import build_manifest, load_manifest

    if args.manifest:
        manifest = load_manifest(args.manifest)
        if not args.audio_root:
            print("--manifest requires --audio-root.", file=sys.stderr)
            return 2
        audio_root = args.audio_root
    else:
        manifest = build_manifest(args.dataset, args.data_root, split=args.split)
        audio_root = args.audio_root or os.path.join(
            args.data_root, get_spec(args.dataset).rel_path
        )
    bank = NoiseBank(args.noise_dir) if args.noise_dir else None
    noises = [n.strip() for n in args.noise.split(",") if n.strip()]
    snrs = [s for s in parse_snrs(args.snrs) if s is not None]
    samples = manifest["samples"][: args.num_samples] if args.num_samples else manifest["samples"]

    for noise in noises:
        for snr in snrs:
            cond = NoiseCondition(noise, snr, args.noise_seed)
            out_dir = os.path.join(args.out_dir, manifest["dataset"], cond.label)
            index = {}
            for s in samples:
                src = os.path.join(audio_root, s.get(args.path_field) or s["audio"])
                chunk = (
                    (s["chunk_start_s"], s["chunk_end_s"]) if "chunk_start_s" in s else None
                )
                rel = f"{_safe(s['key'])}.wav"
                try:
                    render_sample(src, os.path.join(out_dir, rel), s["key"], cond, bank, chunk)
                except Exception as e:
                    print(f"  failed {s['key']}: {e}")
                    continue
                index[s["key"]] = rel
            with open(os.path.join(out_dir, "index.json"), "w") as f:
                json.dump(
                    {
                        "dataset": manifest["dataset"],
                        "condition": cond.to_dict(),
                        "note": "Audio already chunked; predict times relative to file start.",
                        "files": index,
                    },
                    f,
                    indent=1,
                )
            print(f"{cond.label}: {len(index)} files -> {out_dir}")
    return 0


def _system_label(data: Dict, path: str) -> str:
    from .models.registry import MODEL_IS_FORCED_ALIGNER

    meta = data.get("metadata") or {}
    st = meta.get("system_type")
    if st is None:
        prefix = str(data.get("model", "")).split("-")[0]
        st = (
            "forced_aligner"
            if MODEL_IS_FORCED_ALIGNER.get(prefix, False)
            else "transcription"
        )
    dep = meta.get("deployment") or ("hosted_api" if "results_api" in path else "local")
    if st == "forced_aligner":
        return "forced aligner (hosted API)" if dep == "hosted_api" else "forced aligner"
    return "transcription (hosted API)" if dep == "hosted_api" else "transcription (local)"


_CHECKPOINT_DISPLAY_NAMES = {
    "large-v2_verbatimize_longform_with_timing_release": "crisperwhisper_2.0_large",
    "large-v2_verbatimize_longform_pro_with_timing": "crisperwhisper_2.0_large_pro",
}


def _display_name(data: Dict, path: str) -> str:
    from .models.registry import MODEL_FACTORIES

    meta = data.get("metadata") or {}
    if meta.get("display_name"):
        return meta["display_name"]
    model = str(data.get("model") or os.path.basename(path))
    for checkpoint, name in _CHECKPOINT_DISPLAY_NAMES.items():
        if model.endswith(checkpoint):
            return name
    prefix = model.split("-")[0]
    return prefix if prefix in MODEL_FACTORIES else model


def _load_rows(paths: Sequence[str]) -> List[Dict]:
    rows = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        if "overall" not in data:
            continue
        o = data["overall"]
        rows.append(
            {
                "model": _display_name(data, path),
                "dataset": data.get("dataset", ""),
                "condition": NoiseCondition.from_dict(data.get("condition")).label,
                "type": _system_label(data, path),
                "mae": o["mae_ms"]["overall"],
                "medae": (o.get("median_ae_ms") or {}).get("overall"),
                "miou": o["miou"],
                "mediou": o.get("median_iou"),
                "word_recall": o.get("word_recall"),
                "f1": {lbl: m["f1"] for lbl, m in o["f1_standard"].items()},
            }
        )
    return rows


def _markdown_leaderboard(rows: List[Dict], collar_labels: Sequence[str]) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["condition"])].append(r)
    out = []
    for (dataset, cond), rs in sorted(groups.items()):
        rs.sort(key=lambda r: (r["mae"] is None, r["mae"]))
        title = dataset if cond == CLEAN else f"{dataset} — {cond}"
        out.append(f"**{title}**\n")
        head = ["model", "type", "MAE (ms)", "MedAE (ms)", "mIoU", "MedIoU", "word recall"]
        head += [f"F1@{lbl}" for lbl in collar_labels]
        out.append("| " + " | ".join(head) + " |")
        out.append("| " + " | ".join("---" for _ in head) + " |")
        for r in rs:
            cells = [
                f"`{r['model']}`",
                r["type"],
                _fmt(r["mae"]),
                _fmt(r["medae"]),
                _fmt(r["miou"], pct=True),
                _fmt(r["mediou"], pct=True),
                _fmt(r["word_recall"], pct=True),
            ]
            cells += [_fmt(r["f1"].get(lbl), pct=True) for lbl in collar_labels]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


_PIVOT_METRICS = {
    "mae": ("MAE (ms)", False),
    "medae": ("MedAE (ms)", False),
    "miou": ("mIoU", True),
    "mediou": ("MedIoU", True),
    "word_recall": ("word recall", True),
}


def _condition_sort_key(label: str):
    cond = NoiseCondition.from_dict(
        {"noise": CLEAN}
        if label == CLEAN
        else {"noise": label.rsplit("_snr", 1)[0], "snr_db": float(label.rsplit("_snr", 1)[1])}
    )
    return (not cond.is_clean, cond.noise, -(cond.snr_db or 0.0))


def _pivot_value(row: Dict, metric: str):
    if metric.startswith("f1@"):
        return row["f1"].get(metric[3:])
    return row[metric]


def _markdown_pivot(rows: List[Dict], metric: str) -> str:
    title, pct = _PIVOT_METRICS.get(metric, (metric.upper(), metric.startswith("f1@")))
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == dataset]
        conds = sorted({r["condition"] for r in rs}, key=_condition_sort_key)
        table: Dict[tuple, Dict[str, Dict]] = defaultdict(dict)
        for r in rs:
            table[(r["model"], r["type"])][r["condition"]] = r
        out.append(f"**{dataset} — {title}**\n")
        out.append("| model | type | " + " | ".join(conds) + " |")
        out.append("| --- | --- | " + " | ".join("---" for _ in conds) + " |")
        for (model, typ), by_cond in sorted(table.items()):
            cells = [
                _fmt(_pivot_value(by_cond[c], metric), pct=pct) if c in by_cond else ""
                for c in conds
            ]
            out.append(f"| `{model}` | {typ} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def _emit(text: str, output: Optional[str]) -> None:
    if output:
        with open(output, "w") as f:
            f.write(text)
        print(f"Wrote {output}")
    else:
        print(text)


def cmd_compare(args: argparse.Namespace) -> int:
    rows = _load_rows(args.results)
    if not rows:
        print("No result files with an 'overall' section.", file=sys.stderr)
        return 1
    collar_labels = (
        [x.strip() for x in args.collar_labels.split(",")]
        if args.collar_labels
        else list(rows[0]["f1"].keys())
    )

    if args.pivot:
        _emit(_markdown_pivot(rows, args.pivot), args.output)
        return 0
    if args.markdown:
        _emit(_markdown_leaderboard(rows, collar_labels), args.output)
        return 0

    header = (
        f"{'model':<28} {'dataset':<20} {'condition':<16} {'MAE':>7} {'MedAE':>7} "
        f"{'mIoU':>7} {'MedIoU':>7} {'WordRec':>8}"
    )
    for lbl in collar_labels:
        header += f" {('F1@' + lbl):>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['model'][:27]:<28} {str(r['dataset'])[:19]:<20} {r['condition'][:15]:<16} "
            f"{_fmt(r['mae']):>7} {_fmt(r['medae']):>7} {_fmt(r['miou'], pct=True):>7} "
            f"{_fmt(r['mediou'], pct=True):>7} {_fmt(r['word_recall'], pct=True):>8}"
        )
        for lbl in collar_labels:
            line += f" {_fmt(r['f1'].get(lbl), pct=True):>9}"
        print(line)
    return 0


def _add_model_args(p: argparse.ArgumentParser) -> None:
    from .models.registry import list_models

    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--model",
        required=True,
        help=f"Built-in model name. Available: {', '.join(list_models())}.",
    )
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use N evenly spaced samples (sorted keys) instead of the full split.",
    )
    p.add_argument("--collars", default=None, help="Comma-separated seconds.")
    p.add_argument("--out", default=None, help="Directory to write result JSON.")
    p.add_argument(
        "--save-predictions",
        default=None,
        metavar="DIR",
        help=(
            "Directory to write the canonical per-word predictions JSON "
            "(re-scorable via 'score'), like the hosted-API providers."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse predictions already in --save-predictions and only predict "
            "the missing samples (e.g. after API errors)."
        ),
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Where to render sliced/noisy audio (default: system temp).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for adapters that support it (used by crisperwhisper).",
    )
    p.add_argument(
        "--compute-type",
        default=None,
        help="Compute type for adapters that support it (used by crisperwhisper).",
    )
    p.add_argument(
        "--crisperwhisper-model",
        default=None,
        help="HF id, HF-format dir, or CT2 dir for --model crisperwhisper.",
    )
    p.add_argument(
        "--crisperwhisper-backend",
        choices=["transformers", "ct2"],
        default=None,
        help="Backend for --model crisperwhisper.",
    )
    p.add_argument(
        "--crisperwhisper-cache",
        default=None,
        help="CT2 conversion cache for --model crisperwhisper.",
    )
    p.add_argument(
        "--crisperwhisper-repo",
        default=None,
        help="Path containing a local crisperwhisper checkout.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nyra-timing",
        description="Speech timing accuracy benchmark (MAE, MedAE, IoU, F1@xms).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-datasets", help="List registered datasets.")
    p_list.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_list.set_defaults(func=cmd_list_datasets)

    p_man = sub.add_parser(
        "export-manifest", help="Write public, text-free sample manifests."
    )
    p_man.add_argument("--dataset", default="all")
    p_man.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_man.add_argument("--split", default=None)
    p_man.add_argument("--out", default="manifests")
    p_man.set_defaults(func=cmd_export_manifest)

    p_val = sub.add_parser("validate", help="Validate a submission file.")
    p_val.add_argument("--predictions", required=True)
    p_val.add_argument(
        "--manifest", default=None, help="Public manifest (no ground truth needed)."
    )
    p_val.add_argument("--dataset", default=None)
    p_val.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_val.add_argument("--split", default=None)
    p_val.set_defaults(func=cmd_validate)

    p_score = sub.add_parser("score", help="Score predictions file(s).")
    p_score.add_argument(
        "--dataset", default=None, help="Defaults to the 'dataset' field of each file."
    )
    p_score.add_argument(
        "--predictions", required=True, nargs="+", help="Predictions JSON file(s)."
    )
    p_score.add_argument("--model-name", default=None)
    p_score.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_score.add_argument("--split", default=None)
    p_score.add_argument("--collars", default=None, help="Comma-separated seconds.")
    p_score.add_argument(
        "--subset", type=int, default=None, help="Score only N evenly spaced samples."
    )
    p_score.add_argument("--out", default=None, help="Directory to write result JSON.")
    p_score.set_defaults(func=cmd_score)

    p_run = sub.add_parser("run", help="Run a built-in model adapter, then score.")
    _add_model_args(p_run)
    _add_noise_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_rob = sub.add_parser(
        "robustness", help="Sweep a built-in model over noise types and SNRs."
    )
    _add_model_args(p_rob)
    _add_noise_args(p_rob, multi=True)
    p_rob.set_defaults(func=cmd_robustness)

    p_speed = sub.add_parser(
        "speed", help="Measure a built-in model's real-time factor (clean audio)."
    )
    _add_model_args(p_speed)
    p_speed.add_argument("--longest", type=int, default=100, help="Time the N longest clips.")
    p_speed.add_argument(
        "--warmup", type=int, default=2, help="Untimed warm-up clips (the shortest ones)."
    )
    p_speed.set_defaults(func=cmd_speed)

    p_noisy = sub.add_parser(
        "make-noisy", help="Render noisy copies of a dataset for external models."
    )
    p_noisy.add_argument("--dataset", default=None)
    p_noisy.add_argument("--manifest", default=None)
    p_noisy.add_argument(
        "--audio-root",
        default=None,
        help="Root that manifest audio paths are relative to (your corpus copy).",
    )
    p_noisy.add_argument(
        "--path-field",
        default="audio",
        choices=["audio", "corpus_path"],
        help="Manifest field to resolve under --audio-root.",
    )
    p_noisy.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_noisy.add_argument("--split", default=None)
    p_noisy.add_argument("--num-samples", type=int, default=None)
    p_noisy.add_argument("--out-dir", required=True)
    _add_noise_args(p_noisy, multi=True)
    p_noisy.set_defaults(func=cmd_make_noisy)

    p_cmp = sub.add_parser("compare", help="Compare multiple result JSON files.")
    p_cmp.add_argument("results", nargs="+", help="Result JSON files.")
    p_cmp.add_argument(
        "--markdown", action="store_true", help="Emit one markdown table per dataset/condition."
    )
    p_cmp.add_argument(
        "--pivot",
        default=None,
        metavar="METRIC",
        help="Markdown table of models x conditions for one metric: "
        "mae, medae, miou, mediou, word_recall or f1@50ms (etc.).",
    )
    p_cmp.add_argument("--output", default=None, help="Write markdown to this file.")
    p_cmp.add_argument(
        "--collar-labels", default=None, help="Subset of collars, e.g. '50ms,100ms,200ms'."
    )
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "make-noisy" and not (args.dataset or args.manifest):
        parser.error("make-noisy needs --dataset or --manifest")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
