"""MMS-FA adapter (Meta's Massively Multilingual Speech forced aligner).

Uses torchaudio's ``MMS_FA`` bundle (Pratap et al., 2023): a wav2vec2 model
fine-tuned for character-level CTC alignment over 1,100+ languages. Words are
romanised to the bundle's character set, aligned with torchaudio's CTC forced
aligner, and each word spans its first to its last aligned character frame.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from .base import TimingModel


def _normalize(word: str) -> str:
    w = (word or "").strip().lower()
    w = w.rstrip(".,;:!?\"'")
    w = w.lstrip("\"'")
    return w


class MmsFaModel(TimingModel):
    name = "mms_fa"
    requires_transcript = True

    def __init__(self, device: Optional[str] = None):
        try:
            import torch
            from torchaudio.pipelines import MMS_FA
        except ImportError as e:
            raise ImportError("mms_fa needs torch and torchaudio>=2.1") from e
        self.torch = torch
        self.bundle = MMS_FA
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MMS_FA.get_model(with_star=False).to(self.device).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()
        self.charset = set(MMS_FA.get_dict(star=None))

    def _mms_chars(self, word: str) -> str:
        return "".join(c for c in re.sub(r"[^a-z']", "", word) if c in self.charset)

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        import torchaudio

        ref = [_normalize(w) for w in (words or (transcript or "").split())]
        pairs = [(w, self._mms_chars(w)) for w in ref]
        pairs = [(w, c) for w, c in pairs if c]
        if not pairs:
            return []
        wav, sr = torchaudio.load(audio_path)
        if sr != self.bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.bundle.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        with self.torch.inference_mode():
            emission, _ = self.model(wav.to(self.device))
            emission = self.torch.log_softmax(emission, dim=-1)
            seconds_per_frame = wav.shape[1] / self.bundle.sample_rate / emission.shape[1]
            spans = self.aligner(emission[0], self.tokenizer([c for _, c in pairs]))
        return [(w, s[0].start * seconds_per_frame, s[-1].end * seconds_per_frame)
                for (w, _), s in zip(pairs, spans) if s]
