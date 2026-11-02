"""ElevenLabs forced alignment (hosted API).

``POST https://api.elevenlabs.io/v1/forced-alignment`` with a multipart ``file``
and the reference ``text`` returns ``words: [{text, start, end, loss}]`` in
seconds. Reads ``ELEVENLABS_API_KEY`` from the environment.

The reference is sent as space-joined words; non-speech event tokens (Buckeye
``laughter``) are left out because the API has no notion of them and they are
excluded from scoring. Requests run concurrently (``max_workers``) with retries
on rate limits and server errors.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from ..normalize import is_scored_word
from .base import PredictItem, TimingModel

API_URL = "https://api.elevenlabs.io/v1/forced-alignment"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ElevenLabsForcedAligner(TimingModel):
    requires_transcript = True
    name = "elevenlabs_fa"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = API_URL,
        max_workers: int = 8,
        max_retries: int = 6,
        timeout: float = 300.0,
    ):
        import requests

        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self._api_key:
            raise ValueError("ElevenLabsForcedAligner needs ELEVENLABS_API_KEY.")
        self._session = requests.Session()
        self._api_url = api_url
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        tokens = [w.strip() for w in (words or []) if is_scored_word(w)]
        text = " ".join(tokens) if words is not None else (transcript or "")
        if not text.strip():
            return []
        data = self._post(audio_path, text)
        out: List[Tuple[str, float, float]] = []
        for w in data.get("words") or []:
            t = (w.get("text") or "").strip()
            s, e = w.get("start"), w.get("end")
            if t and s is not None and e is not None:
                out.append((t, float(s), float(e)))
        return out

    def _post(self, audio_path: str, text: str) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with open(audio_path, "rb") as f:
                    r = self._session.post(
                        self._api_url,
                        headers={"xi-api-key": self._api_key},
                        data={"text": text},
                        files={"file": (os.path.basename(audio_path), f)},
                        timeout=self.timeout,
                    )
            except Exception:
                if attempt == self.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in _RETRY_STATUS or attempt == self.max_retries:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 100
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        total = len(items)
        with ThreadPoolExecutor(self.max_workers) as pool:
            futures = {
                pool.submit(self.predict, it.audio_path, it.transcript, it.words): it.key
                for it in items
            }
            for n, fut in enumerate(as_completed(futures), start=1):
                key = futures[fut]
                try:
                    out[key] = fut.result()
                except Exception as e:
                    print(f"  error on {key}: {e}")
                if progress_every and (n % progress_every == 0 or n == 1 or n == total):
                    print(f"  [{n}/{total}] done ({len(out)} ok)", flush=True)
        return out
