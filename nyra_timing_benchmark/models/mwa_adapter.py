"""MWA (Multilingual Word Aligner) adapter.

MWA (https://github.com/MLSpeech/Multilingual-Word-Aligner, Weber et al.,
Interspeech 2026) combines MMS-FA emissions, UnsupSeg boundary features and a
Conformer boundary classifier, decoded with dynamic programming. It predicts one
boundary between consecutive words, so every word starts where the previous one
ends and the first word starts at 0 s.

MWA pins old ``transformers``/``huggingface_hub`` versions, so it runs in its
own environment: a long-lived worker process (``mwa_worker.py``) loads the
models once and aligns one clip per request. Configure with ``MWA_PYTHON`` (the
environment's interpreter) and ``MWA_REPO`` (a checkout of the MWA repository).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel

TARGET_SR = 16_000
WORKER = Path(__file__).with_name("mwa_worker.py")


def _normalize(word: str) -> str:
    w = (word or "").strip().lower()
    w = w.rstrip(".,;:!?\"'")
    w = w.lstrip("\"'")
    return w


class MwaModel(TimingModel):
    requires_transcript = True

    def __init__(
        self,
        model_name: str = "buckeye",
        python: Optional[str] = None,
        repo: Optional[str] = None,
        device: str = "cuda:0",
        workers: Optional[int] = None,
    ):
        self.python = python or os.environ.get("MWA_PYTHON", "/home/ubuntu/mwa_venv/bin/python")
        self.repo = repo or os.environ.get(
            "MWA_REPO", "/home/ubuntu/data3/Multilingual-Word-Aligner")
        for path, what in ((self.python, "MWA_PYTHON"), (self.repo, "MWA_REPO")):
            if not os.path.exists(path):
                raise ImportError(f"{what} not found at {path}; see the MWA install notes.")
        self.name = f"mwa-{model_name}"
        self.model_name = model_name
        self.device = device
        self.workers = workers or int(os.environ.get("MWA_WORKERS", "1"))
        self._tmp = Path(tempfile.mkdtemp(prefix="nyra_mwa_"))
        env = dict(os.environ, HF_HUB_OFFLINE=os.environ.get("MWA_HF_HUB_OFFLINE", "0"))
        if self.workers > 1:
            env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        self._procs = [
            subprocess.Popen(
                [self.python, str(WORKER), self.repo, model_name, device],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1,
            )
            for _ in range(self.workers)
        ]
        for proc in self._procs:
            ready = proc.stdout.readline()
            if not ready or not json.loads(ready).get("ready"):
                raise RuntimeError("MWA worker failed to start (see stderr).")
        self._idle: "queue.Queue[int]" = queue.Queue()
        for i in range(self.workers):
            self._idle.put(i)

    def _prep_wav(self, src: str, dst: Path) -> None:
        import torchaudio

        wav, sr = torchaudio.load(src)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        torchaudio.save(str(dst), wav, TARGET_SR)

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        ref = [_normalize(w) for w in (words or (transcript or "").split())]
        pairs = [(w, re.sub(r"[^a-z']", "", w)) for w in ref]
        pairs = [(w, c) for w, c in pairs if c]
        if not pairs:
            return []
        i = self._idle.get()
        try:
            wav = self._tmp / f"utt_{i}.wav"
            self._prep_wav(audio_path, wav)
            proc = self._procs[i]
            proc.stdin.write(json.dumps({"audio": str(wav), "words": [c for _, c in pairs]}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        finally:
            self._idle.put(i)
        if not line:
            raise RuntimeError("MWA worker exited")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"MWA failed: {resp['error']}")
        if len(resp["words"]) != len(pairs):
            raise RuntimeError("MWA returned a different number of words")
        return [(w, float(s), float(e)) for (w, _), (_, s, e) in zip(pairs, resp["words"])]

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 25
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        if self.workers == 1:
            return super().predict_many(items, progress_every)
        out: Dict[str, List[Tuple[str, float, float]]] = {}

        def run(item: PredictItem):
            return item.key, self.predict(item.audio_path, item.transcript, item.words)

        with ThreadPoolExecutor(self.workers) as pool:
            futures = [pool.submit(run, item) for item in items]
            for n, fut in enumerate(as_completed(futures), 1):
                try:
                    key, words = fut.result()
                    out[key] = words
                except Exception as e:
                    print(f"  [{n}/{len(items)}] error: {e}")
                if progress_every and n % progress_every == 0:
                    print(f"  [{n}/{len(items)}] done")
        return out

    def close(self) -> None:
        for proc in getattr(self, "_procs", []):
            if proc.poll() is None:
                proc.stdin.close()
                proc.terminate()
        shutil.rmtree(getattr(self, "_tmp", ""), ignore_errors=True)

    def __del__(self):
        self.close()
