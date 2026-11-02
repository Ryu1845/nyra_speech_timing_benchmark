"""Qwen3 forced-aligner adapter.

Uses ``Qwen/Qwen3-ForcedAligner-0.6B`` via the ``qwen_asr`` package to align the
reference text to audio. Requires ``qwen_asr`` (and a compatible ``transformers``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel


def _normalize(word: str) -> str:
    w = (word or "").strip().lower()
    w = w.rstrip(".,;:!?\"'")
    w = w.lstrip("\"'")
    return w


def _patch_check_model_inputs() -> None:
    """Make ``transformers.utils.generic.check_model_inputs`` accept the no-arg
    factory form that ``qwen_asr 0.0.6`` uses (``@check_model_inputs()``).

    Several transformers releases define it as ``check_model_inputs(func)`` (a
    bare decorator), which breaks qwen_asr's factory-style call. This wrapper
    supports both call styles and is idempotent.
    """
    try:
        import transformers.utils.generic as g
    except Exception:
        return
    orig = getattr(g, "check_model_inputs", None)
    if orig is None or getattr(orig, "_nyra_dual", False):
        return

    def patched(func=None, *args, **kwargs):
        if callable(func) and not kwargs:
            return orig(func)

        def deco(f):
            return orig(f)

        return deco

    patched._nyra_dual = True
    g.check_model_inputs = patched


class QwenForcedAligner(TimingModel):
    requires_transcript = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "cuda:0",
        language: str = "English",
    ):
        try:
            import torch

            _patch_check_model_inputs()
            from qwen_asr import Qwen3ForcedAligner
        except Exception as e:  # pragma: no cover - optional/conflicting dep
            raise ImportError(
                "qwen_asr could not be imported for the Qwen forced aligner "
                f"({type(e).__name__}: {e}). Install/repair with `pip install "
                "qwen-asr` in an environment with a compatible transformers."
            ) from e

        self.name = f"qwen_fa-{model_id.split('/')[-1]}"
        self.language = language
        self.batch_size = 8
        self._aligner = Qwen3ForcedAligner.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )

    @staticmethod
    def _text_for(transcript: Optional[str], words: Optional[Sequence[str]]) -> str:
        if words:
            return " ".join(_normalize(w) for w in words if _normalize(w))
        if transcript:
            return transcript
        raise ValueError("QwenForcedAligner requires reference words or transcript.")

    @staticmethod
    def _to_triples(pred) -> List[Tuple[str, float, float]]:
        out: List[Tuple[str, float, float]] = []
        for w in pred or []:
            text_w = _normalize(getattr(w, "text", ""))
            if text_w:
                out.append((text_w, float(w.start_time), float(w.end_time)))
        return out

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        text = self._text_for(transcript, words)
        result = self._aligner.align(
            audio=audio_path, text=text, language=self.language
        )
        return self._to_triples(result[0] if result else [])

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 25
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        total = len(items)
        bs = max(1, self.batch_size)
        for start in range(0, total, bs):
            batch = list(items[start : start + bs])
            audio = [it.audio_path for it in batch]
            texts = [self._text_for(it.transcript, it.words) for it in batch]
            langs = [self.language] * len(batch)
            try:
                results = self._aligner.align(audio=audio, text=texts, language=langs)
            except Exception as e:
                print(f"  batch align failed ({e}); falling back to per-item")
                results = []
                for it, t in zip(batch, texts):
                    try:
                        one = self._aligner.align(
                            audio=it.audio_path, text=t, language=self.language
                        )
                        results.append(one[0] if one else [])
                    except Exception as e2:
                        print(f"  error on {it.key}: {e2}")
                        results.append([])
            for it, pred in zip(batch, results):
                out[it.key] = self._to_triples(pred)
            done = min(start + bs, total)
            if progress_every:
                print(f"  [{done}/{total}] aligned")
        return out
