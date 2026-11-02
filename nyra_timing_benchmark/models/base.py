"""Model adapter interface.

A timing model takes an audio file and returns word-level timings. Two kinds of
models are supported:

* **ASR models** (e.g. WhisperX, Canary) transcribe freely and emit their own
  word timestamps; they ignore the ``transcript``/``words`` hints.
* **Forced aligners** (e.g. NeMo FA, CTC-segmentation, MFA, Qwen FA) require the
  reference words and only predict their timings. They set
  ``requires_transcript = True`` and consume the ``words`` argument.

Implement :meth:`predict` and register the model in ``registry.py`` to make it
available to the ``run`` command.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PredictItem:
    """One unit of work for batch prediction.

    ``audio_path`` is the file to run inference on (already sliced for chunked
    datasets). ``words``/``transcript`` are the reference text for forced aligners.
    """

    key: str
    audio_path: str
    words: List[str] = field(default_factory=list)
    transcript: str = ""


class TimingModel(ABC):
    """Base class for models that produce word-level timings."""

    #: Short name used in result filenames and tables.
    name: str = "model"

    #: True for forced aligners that need the reference words to align.
    requires_transcript: bool = False

    @abstractmethod
    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        """Return ``(word, start_s, end_s)`` tuples for an audio file.

        ``transcript`` (full text) and ``words`` (token list, 1:1 with the
        reference timings) are provided when available. Forced aligners must use
        them; free ASR models may ignore them.
        """
        raise NotImplementedError

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 25
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """Predict timings for a list of items.

        The default implementation loops :meth:`predict` one item at a time,
        skipping items that raise. Models that align more efficiently in bulk
        (e.g. MFA over a whole corpus) should override this.
        """
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        total = len(items)
        for i, item in enumerate(items):
            try:
                if self.requires_transcript:
                    out[item.key] = self.predict(
                        item.audio_path,
                        transcript=item.transcript,
                        words=item.words,
                    )
                else:
                    out[item.key] = self.predict(item.audio_path)
            except Exception as e:
                print(f"  [{i + 1}/{total}] error on {item.key}: {e}")
                continue
            if progress_every and ((i + 1) % progress_every == 0 or i == 0):
                print(f"  [{i + 1}/{total}] {len(out[item.key])} words from {item.key}")
        return out

    def close(self) -> None:
        """Optional hook to free resources after a run."""
        return None
