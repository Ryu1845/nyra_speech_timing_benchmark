"""AssemblyAI pre-recorded transcription with word timestamps.

Two configurations:

* ``assemblyai_u35_pro`` — ``universal-3-5-pro`` (current best async model) with
  the verbatim prompt of the verbatim-speech benchmark. As of November 2026 the
  3.x models treat ``prompt`` as context only and drop fillers regardless.
* ``assemblyai_u2_disfl`` — ``universal-2`` with ``disfluencies=True``, the only
  AssemblyAI option that keeps fillers (``um``/``uh``) in the transcript.

``words[].start``/``end`` are milliseconds. Reads ``ASSEMBLYAI_API_KEY`` (env or
research ``.env``).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel

_GEN_DIR = Path(__file__).resolve().parents[2] / "generation"

VERBATIM_PROMPT = """Transcribe verbatim:
- Fillers: yes (um, uh, like, you know)
- Repetitions: yes (I I, the the)
- False starts: yes (I was I went)
- Stutters: yes (th- that, b- but)
- Vocal sounds: yes ([laughter], [cough], [breath], [throatclearing], [lipsmack], [sigh], [sniff])"""

CONFIGS = {
    "assemblyai_u35_pro": dict(speech_models=["universal-3-5-pro"], prompt=VERBATIM_PROMPT),
    "assemblyai_u2_disfl": dict(speech_models=["universal-2"], disfluencies=True),
}


class AssemblyAIModel(TimingModel):
    requires_transcript = False

    def __init__(self, variant: str, max_workers: int = 32, max_retries: int = 3):
        import assemblyai as aai

        if str(_GEN_DIR) not in sys.path:
            sys.path.insert(0, str(_GEN_DIR))
        from common import load_env_file, require_key

        load_env_file()
        aai.settings.api_key = require_key("ASSEMBLYAI_API_KEY")
        self._aai = aai
        self._config = aai.TranscriptionConfig(
            language_code="en", punctuate=True, **CONFIGS[variant]
        )
        self.name = variant
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.raw_responses: Dict[str, object] = {}

    def _call(self, audio_path: str) -> Tuple[List[Tuple[str, float, float]], dict]:
        aai = self._aai
        for attempt in range(self.max_retries + 1):
            t = aai.Transcriber(config=self._config).transcribe(audio_path)
            if t.status != aai.TranscriptStatus.error:
                words = [
                    (w.text.strip(), w.start / 1000.0, w.end / 1000.0)
                    for w in (t.words or [])
                    if w.text and w.text.strip() and w.start is not None and w.end is not None
                ]
                raw = t.json_response or {}
                return words, {
                    k: raw.get(k)
                    for k in ("id", "text", "words", "speech_model_used", "audio_duration")
                }
            if attempt == self.max_retries:
                raise RuntimeError(t.error)
            time.sleep(2 * (attempt + 1))
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
