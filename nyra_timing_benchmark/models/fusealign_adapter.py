"""FuseAlign adapter (Zyphra, ``fusealign-multilingual``).

FuseAlign (Vaidya et al., 2026, arXiv 2609.33650) is not a CTC model: a joint
audio/text transformer reads the waveform and the UTF-8 bytes of the transcript
and predicts each word's start and end on a 1 ms grid plus a presence
probability. Words under 0.1 presence take the paper's uniform fallback between
their placed neighbours; every word is returned, as the package returns it.

Uses ``fusealign_hub`` (https://github.com/Zyphra/fusealign-multilingual) with its
default checkpoint, ``fusealign-multilingual-cjkft-step2500.safetensors`` from the
Hugging Face repo ``Zyphra/fusealign``. Reference words are normalized and joined
by spaces, so a reference token with an internal space is aligned as separate
words. Clips over 30 s go through the package's long-form path, which runs
Whisper large-v3 once per clip for anchors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..normalize import normalize_word
from .base import PredictItem, TimingModel

#: Clips per ``align_batch`` call; the package splits a call into padded forwards
#: of at most ``batch_size`` clips.
_GROUP = 256


class FuseAlignModel(TimingModel):
    requires_transcript = True

    def __init__(self, device: Optional[str] = None, batch_size: int = 64):
        try:
            import soundfile
            from fusealign_hub import load_aligner
            from fusealign_hub.aligner import CHECKPOINT_FILE
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "fusealign_hub could not be imported "
                f"({type(e).__name__}: {e}). `pip install -e '.[fusealign]'`."
            ) from e
        self._sf = soundfile
        self.name = CHECKPOINT_FILE.removesuffix(".safetensors")
        self._aligner = load_aligner(device=device, batch_size=batch_size)

    @staticmethod
    def _text(transcript: Optional[str], words: Optional[Sequence[str]]) -> str:
        tokens = (normalize_word(w) for w in (words or (transcript or "").split()))
        text = " ".join(t for t in tokens if t)
        if not text:
            raise ValueError("FuseAlignModel requires reference words or transcript.")
        return text

    def _align(self, pairs: Sequence[Tuple[str, str]]) -> List[List[Tuple[str, float, float]]]:
        """Align ``(audio_path, text)`` pairs, one ``align_batch`` call per sample rate."""
        loaded = []
        for path, _ in pairs:
            wav, sr = self._sf.read(path, dtype="float32", always_2d=True)
            loaded.append((wav.mean(axis=1), sr))
        out: List[List[Tuple[str, float, float]]] = [[] for _ in pairs]
        for sr in {sr for _, sr in loaded}:
            idx = [i for i, (_, r) in enumerate(loaded) if r == sr]
            results = self._aligner.align_batch(
                [loaded[i][0] for i in idx], sr, [pairs[i][1] for i in idx]
            )
            for i, words in zip(idx, results):
                out[i] = [(w.word, float(w.start), float(w.end)) for w in words]
        return out

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        return self._align([(audio_path, self._text(transcript, words))])[0]

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 25
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        total = len(items)
        for start in range(0, total, _GROUP):
            group = list(items[start : start + _GROUP])
            try:
                pairs = [(it.audio_path, self._text(it.transcript, it.words)) for it in group]
                for it, pred in zip(group, self._align(pairs)):
                    out[it.key] = pred
            except Exception as e:
                print(f"  group align failed ({e}); falling back to per-item")
                for it in group:
                    try:
                        out[it.key] = self.predict(it.audio_path, it.transcript, it.words)
                    except Exception as e2:
                        print(f"  error on {it.key}: {e2}")
            if progress_every:
                print(f"  [{min(start + _GROUP, total)}/{total}] aligned")
        return out
