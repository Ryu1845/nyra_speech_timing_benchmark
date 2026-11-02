"""NVIDIA Canary-1b-v2 ASR adapter with word-level timestamps.

Canary transcribes freely and emits word timestamps (it is not a forced aligner),
so it ignores the reference words. Requires ``nemo_toolkit[asr]``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .base import TimingModel


class CanaryModel(TimingModel):
    requires_transcript = False

    def __init__(
        self,
        pretrained_name: str = "nvidia/canary-1b-v2",
        device: str = "cuda",
        source_lang: str = "en",
    ):
        try:
            import torch
            from nemo.collections.asr.models import EncDecMultiTaskModel
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "NeMo is required for Canary. Install with `pip install "
                "nemo_toolkit[asr]`."
            ) from e

        self._torch = torch
        self.name = f"canary-{pretrained_name.split('/')[-1]}"
        self.source_lang = source_lang
        map_location = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model = EncDecMultiTaskModel.from_pretrained(
            pretrained_name, map_location=map_location
        )
        self._model.eval()

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        output = self._model.transcribe(
            [audio_path],
            source_lang=self.source_lang,
            target_lang=self.source_lang,
            timestamps=True,
            batch_size=1,
        )
        if not output:
            return []
        result = output[0]
        out: List[Tuple[str, float, float]] = []
        stamps = getattr(result, "timestamp", None)
        if not stamps:
            return out

        if stamps.get("word"):
            for s in stamps["word"]:
                w = (s.get("word") or "").strip()
                start, end = s.get("start"), s.get("end")
                if w and start is not None and end is not None:
                    out.append((w, float(start), float(end)))
        elif stamps.get("segment"):
            for s in stamps["segment"]:
                text = (s.get("segment") or "").strip()
                start, end = s.get("start"), s.get("end")
                if text and start is not None and end is not None:
                    parts = text.split()
                    if parts:
                        step = (end - start) / len(parts)
                        for i, w in enumerate(parts):
                            out.append((w, start + i * step, start + (i + 1) * step))
        return out

    def close(self) -> None:  # pragma: no cover - cleanup
        import gc

        try:
            del self._model
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass
