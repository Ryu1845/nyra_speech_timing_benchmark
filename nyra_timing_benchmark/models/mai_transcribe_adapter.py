"""Microsoft MAI-Transcribe-2 via OpenRouter.

``POST https://openrouter.ai/api/v1/audio/transcriptions`` with
``model=microsoft/mai-transcribe-2``, base64 audio, ``response_format=
verbose_json`` and ``timestamp_granularities=["word"]`` returns
``words: [{word, start, end}]`` (seconds). The Azure passthrough option
``enhancedMode.modelOptions.transcribeStyle="verbatim"`` keeps fillers and false
starts, which the benchmark scores (fillers are canonicalised by
``normalize_word``). Reads ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel

API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class MaiTranscribeModel(TimingModel):
    requires_transcript = False

    def __init__(
        self,
        model: str = "microsoft/mai-transcribe-2",
        language: str = "en",
        style: str = "verbatim",
        max_workers: int = 8,
        max_retries: int = 5,
        timeout: float = 300.0,
    ):
        import requests

        self._key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self._key:
            raise ValueError("MaiTranscribeModel needs OPENROUTER_API_KEY.")
        self._session = requests.Session()
        self.model = model
        self.language = language
        self.style = style
        self.name = "mai_transcribe_2"
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.raw_responses: Dict[str, object] = {}

    def _payload(self, audio_path: str) -> dict:
        with open(audio_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        fmt = os.path.splitext(audio_path)[1].lstrip(".").lower() or "wav"
        return {
            "model": self.model,
            "input_audio": {"data": data, "format": fmt},
            "language": self.language,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
            "provider": {
                "options": {
                    "azure": {
                        "enhancedMode": {"modelOptions": {"transcribeStyle": self.style}}
                    }
                }
            },
        }

    def _call(self, audio_path: str) -> Tuple[List[Tuple[str, float, float]], dict]:
        payload = self._payload(audio_path)
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=payload,
                    timeout=self.timeout,
                )
            except Exception:
                if attempt == self.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    data = r.json()
                    if "error" in data and "text" not in data:
                        raise RuntimeError(str(data["error"])[:300])
                    words = []
                    for w in data.get("words") or []:
                        t = (w.get("word") or "").strip()
                        s, e = w.get("start"), w.get("end")
                        if t and s is not None and e is not None:
                            words.append((t, float(s), float(e)))
                    return words, data
                if r.status_code not in _RETRY_STATUS or attempt == self.max_retries:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
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
