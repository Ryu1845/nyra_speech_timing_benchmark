"""Fish Audio ``transcribe-1-pro`` ASR with word-level timestamps.

``POST https://api.fish.audio/v1/asr`` with the ``model: transcribe-1-pro``
header (a missing/mismatched header silently falls back to ``transcribe-1``).
``segments`` are word-level ``{text, start, end}`` in seconds, without
punctuation, speaker markers or event cues. Reads ``FISH_API_KEY`` or
``FISH_AUDIO_API_KEY``.
"""

from __future__ import annotations

import mimetypes
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from .base import PredictItem, TimingModel

API_URL = "https://api.fish.audio/v1/asr"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class FishAsrModel(TimingModel):
    requires_transcript = False

    def __init__(
        self,
        model: str = "transcribe-1-pro",
        language: str = "en",
        max_workers: int = 5,
        max_retries: int = 10,
        api_key: Optional[str] = None,
    ):
        key = api_key or os.environ.get("FISH_API_KEY") or os.environ.get("FISH_AUDIO_API_KEY")
        if not key:
            raise RuntimeError("Set FISH_API_KEY (or FISH_AUDIO_API_KEY)")
        self._key = key
        self.model = model
        self.language = language
        self.name = "fish_" + model.replace("-", "_")
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.raw_responses: Dict[str, object] = {}

    def _call(self, audio_path: str) -> Tuple[List[Tuple[str, float, float]], dict]:
        path = Path(audio_path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        audio = path.read_bytes()
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {self._key}", "model": self.model},
                    files={"audio": (path.name, audio, mime)},
                    data={
                        "language": self.language,
                        "ignore_timestamps": "false",
                        "diarize": "false",
                    },
                    timeout=900,
                )
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
                continue
            if r.status_code in _RETRY_STATUS and attempt < self.max_retries:
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            words = [
                (s["text"].strip(), float(s["start"]), float(s["end"]))
                for s in data.get("segments") or []
                if s.get("text") and s["text"].strip()
            ]
            return words, data
        raise RuntimeError("unreachable")

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        return self._call(audio_path)[0]

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 100
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        self.raw_responses = {}
        total = len(items)
        with ThreadPoolExecutor(self.max_workers) as pool:
            futures = {pool.submit(self._call, it.audio_path): it.key for it in items}
            for n, fut in enumerate(as_completed(futures), start=1):
                key = futures[fut]
                try:
                    words, raw = fut.result()
                    out[key] = words
                    self.raw_responses[key] = raw
                except Exception as e:
                    print(f"  error on {key}: {type(e).__name__}: {e}")
                    self.raw_responses[key] = {"error": f"{type(e).__name__}: {e}"}
                if progress_every and (n % progress_every == 0 or n == 1 or n == total):
                    print(f"  [{n}/{total}] done ({len(out)} ok)", flush=True)
        return out
