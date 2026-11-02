"""NVIDIA Nemotron 3.5 Transcribe (``nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B``).

The model runs in a local vLLM server with the NeMo Speech plugin and its CTC
timestamp middleware (see the README for the serve command); this adapter is a
client of that server's OpenAI-compatible ``/v1/chat/completions`` endpoint.
Each clip is sent with NVIDIA's verbatim transcription prompt, greedy decoding,
reasoning off and ``capture_ctc_timestamps``; the word times come from the
response's ``ctc_timestamps.words``. Speaker and event tags in the transcript
are not words and are dropped.

``NEMOTRON_URL`` sets the server (default ``http://127.0.0.1:8011``) and
``NEMOTRON_MODEL`` the served model name.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel

VERBATIM_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers "
    "and repetitions. Formatting: punctuation and capitalization. Output: transcript only."
)
DEFAULT_URL = "http://127.0.0.1:8011"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
_TAG = re.compile(r"^<[^>]+>$")


class NemotronTranscribeModel(TimingModel):
    requires_transcript = False
    name = "nemotron_transcribe"

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        max_workers: int = 16,
        max_retries: int = 3,
        timeout: float = 600.0,
    ):
        import requests

        self._session = requests.Session()
        self._url = (url or os.environ.get("NEMOTRON_URL", DEFAULT_URL)).rstrip("/")
        self.model = model or os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.raw_responses: Dict[str, object] = {}

    def _body(self, audio_path: str) -> dict:
        path = Path(audio_path)
        mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
        audio = base64.b64encode(path.read_bytes()).decode()
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": VERBATIM_PROMPT},
                {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{audio}"}},
            ]}],
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "mm_processor_kwargs": {"capture_ctc_timestamps": True},
        }

    def _call(self, audio_path: str) -> Tuple[List[Tuple[str, float, float]], dict]:
        body = self._body(audio_path)
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(f"{self._url}/v1/chat/completions", json=body, timeout=self.timeout)
            except Exception:
                if attempt == self.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    break
                if r.status_code < 500 or attempt == self.max_retries:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(2.0 * (attempt + 1))
        data = r.json()
        ts = data.get("ctc_timestamps") or {}
        if ts.get("error") is not None:
            raise RuntimeError(f"timestamp alignment failed: {ts['error']}")
        words = [
            (w["word"].strip(), float(w["start"]), float(w["end"]))
            for w in ts.get("words") or []
            if w.get("word") and w["word"].strip() and not _TAG.match(w["word"].strip())
        ]
        raw = {
            "text": data["choices"][0]["message"]["content"],
            "ctc_timestamps": ts,
            "usage": data.get("usage"),
        }
        return words, raw

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
