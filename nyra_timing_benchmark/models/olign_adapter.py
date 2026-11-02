"""Olign 1.0 forced alignment (Olewave, hosted API).

``POST https://api.olewave.com/olign/v1`` with multipart ``audio`` and
``transcript`` submits a job (``202 {"job_id"}``); ``GET .../v1/<job_id>`` is
polled until ``state`` is ``done``, when ``result.words[]`` holds ``word``,
``begin`` and ``end`` in seconds. Reads ``OLIGN_KEY`` (or ``OLIGN_API_KEY``)
from the environment. See https://github.com/olewave/olign-client.

The reference is sent as space-joined words without non-speech event tokens
(Buckeye ``laughter``), which are excluded from scoring. Words Olign marks as
not spoken come back with zero duration and are kept as returned; ``[tag]``
annotations are dropped. Jobs run concurrently (``max_workers``) with retries
on rate limits, server errors and lost jobs.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..normalize import is_scored_word
from .base import PredictItem, TimingModel

API_URL = "https://api.olewave.com/olign/v1"
USER_AGENT = "nyra-speech-timing-benchmark/1.0"
_RETRY_STATUS = {404, 408, 409, 425, 429, 500, 502, 503, 504}


class OlignModel(TimingModel):
    requires_transcript = True
    name = "olign_v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = API_URL,
        max_workers: int = 8,
        max_retries: int = 6,
        poll_interval: float = 1.0,
        job_timeout: float = 900.0,
    ):
        import requests

        self._key = api_key or os.environ.get("OLIGN_KEY") or os.environ.get("OLIGN_API_KEY", "")
        if not self._key:
            raise ValueError("OlignModel needs OLIGN_KEY.")
        self._requests = requests
        self._api_url = api_url
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self.raw_responses: Dict[str, object] = {}

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self._key, "User-Agent": USER_AGENT}

    def _run_job(self, audio_path: str, text: str) -> dict:
        path = Path(audio_path)
        r = self._requests.post(
            self._api_url,
            headers=self._headers(),
            files={"audio": (path.name, path.read_bytes()),
                   "transcript": ("transcript.txt", text.encode())},
            timeout=300,
        )
        if r.status_code != 202:
            raise _HttpError(r.status_code, r.text[:300])
        job = r.json()["job_id"]
        deadline = time.time() + self.job_timeout
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            st = self._requests.get(f"{self._api_url}/{job}", headers=self._headers(), timeout=60)
            if st.status_code != 200:
                raise _HttpError(st.status_code, st.text[:300])
            data = st.json()
            if data.get("state") == "done":
                data["_versions"] = {k: v for k, v in st.headers.items()
                                     if k.lower().startswith("x-olign-")}
                return data
            if data.get("state") == "failed":
                raise RuntimeError(f"job failed: {data.get('log_tail', '')[-300:]}")
        raise TimeoutError(f"job {job} not done after {self.job_timeout:.0f} s")

    def _call(self, audio_path: str, text: str) -> Tuple[List[Tuple[str, float, float]], dict]:
        for attempt in range(self.max_retries + 1):
            try:
                data = self._run_job(audio_path, text)
                break
            except (_HttpError, self._requests.RequestException, TimeoutError) as e:
                status = getattr(e, "status", None)
                if attempt == self.max_retries or (status is not None and status not in _RETRY_STATUS):
                    raise
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        words = [
            (w["word"], float(w["begin"]), float(w["end"]))
            for w in (data.get("result") or {}).get("words") or []
            if w.get("word") and not w.get("annotation")
        ]
        return words, data

    @staticmethod
    def _text(transcript: Optional[str], words: Optional[Sequence[str]]) -> str:
        if words is not None:
            return " ".join(w.strip() for w in words if is_scored_word(w))
        return transcript or ""

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        text = self._text(transcript, words)
        return self._call(audio_path, text)[0] if text.strip() else []

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 100
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        self.raw_responses = {}
        total = len(items)
        todo = [(it, self._text(it.transcript, it.words)) for it in items]
        with ThreadPoolExecutor(self.max_workers) as pool:
            futures = {pool.submit(self._call, it.audio_path, text): it.key
                       for it, text in todo if text.strip()}
            for n, fut in enumerate(as_completed(futures), start=1):
                key = futures[fut]
                try:
                    words, raw = fut.result()
                    out[key] = words
                    self.raw_responses[key] = raw
                except Exception as e:
                    print(f"  error on {key}: {type(e).__name__}: {e}")
                    self.raw_responses[key] = {"error": f"{type(e).__name__}: {e}"}
                if progress_every and (n % progress_every == 0 or n == 1 or n == len(futures)):
                    print(f"  [{n}/{total}] done ({len(out)} ok)", flush=True)
        return out


class _HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
