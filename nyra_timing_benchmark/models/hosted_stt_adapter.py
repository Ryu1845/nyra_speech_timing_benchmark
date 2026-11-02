"""Hosted speech-to-text APIs as model adapters.

Wraps the ``build_transcriber`` functions of the provider scripts in
``generation/`` so hosted STT systems run through ``nyra-timing run`` /
``robustness`` (noise conditions, Buckeye slicing, subsets) like local models.
Keys come from the environment or the research ``.env`` (see
``generation/common.py``). Requests run concurrently with retries; raw provider
responses are kept in ``raw_responses`` for auditing.
"""

from __future__ import annotations

import importlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel

_GEN_DIR = Path(__file__).resolve().parents[2] / "generation"

#: provider id -> (env var, extra build_transcriber args after the key)
PROVIDERS = {
    "xai_grok_stt": ("XAI_API_KEY", ("en",)),
    "deepgram_nova3": ("DEEPGRAM_API_KEY", ("nova-3", "en")),
    "cartesia_ink_whisper": ("CARTESIA_API_KEY", ("ink-whisper", "en")),
}


class HostedSttModel(TimingModel):
    requires_transcript = False

    def __init__(self, provider: str, max_workers: int = 8, max_retries: int = 5):
        if provider not in PROVIDERS:
            raise KeyError(f"Unknown hosted provider '{provider}'.")
        if str(_GEN_DIR) not in sys.path:
            sys.path.insert(0, str(_GEN_DIR))
        common = importlib.import_module("common")
        script = importlib.import_module(f"generate_{provider}")
        common.load_env_file()
        env_var, extra = PROVIDERS[provider]
        api_key = common.require_key(env_var)
        self._transcribe = script.build_transcriber(api_key, *extra)
        self.name = provider
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.raw_responses: Dict[str, object] = {}

    def _call(self, audio_path: str):
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                return self._transcribe(Path(audio_path))
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        triples, _ = self._call(audio_path)
        return [(w, float(s), float(e)) for w, s, e in triples]

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
                    triples, raw = fut.result()
                except Exception as e:
                    print(f"  error on {key}: {type(e).__name__}: {e}")
                    self.raw_responses[key] = {"error": f"{type(e).__name__}: {e}"}
                    continue
                out[key] = [(w, float(s), float(e)) for w, s, e in triples]
                self.raw_responses[key] = raw
                if progress_every and (n % progress_every == 0 or n == 1 or n == total):
                    print(f"  [{n}/{total}] done ({len(out)} ok)", flush=True)
        return out
