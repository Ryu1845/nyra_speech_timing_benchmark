"""WhisperX timing-model adapter.

Uses wav2vec2 forced alignment via the optional ``whisperx`` package to produce
word-level timestamps. Install with ``pip install whisperx`` (or the
``[whisperx]`` extra). Heavy imports are deferred to construction time so the
rest of the benchmark works without the dependency.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .base import TimingModel


def _allow_pyannote_checkpoint_load() -> None:
    """Work around PyTorch >=2.6 defaulting ``torch.load`` to ``weights_only=True``.

    Pyannote's VAD checkpoint (used by WhisperX) pickles ``omegaconf`` objects,
    which the safe loader rejects. We allowlist the needed globals and, as a
    fallback for trusted local checkpoints, force ``weights_only=False`` so the
    VAD model loads. This only affects WhisperX/pyannote model loading.
    """
    try:
        import torch
    except Exception:
        return

    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode

        torch.serialization.add_safe_globals(
            [ListConfig, DictConfig, ContainerMetadata, Metadata, AnyNode]
        )
    except Exception:
        pass

    if getattr(torch.load, "_nyra_wo_patched", False):
        return
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    _patched_load._nyra_wo_patched = True
    torch.load = _patched_load


def _extract_word_timings(result: dict) -> List[Tuple[str, float, float]]:
    """Flatten WhisperX segments into ``(word, start, end)`` tuples."""
    out: List[Tuple[str, float, float]] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if word and start is not None and end is not None:
                out.append((word, float(start), float(end)))
    return out


class WhisperXModel(TimingModel):
    name = "whisperx"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
        language: str = "en",
    ):
        try:
            import whisperx  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "whisperx is required for the WhisperX adapter. "
                "Install it with `pip install whisperx`."
            ) from e

        self._whisperx = whisperx
        self.device = device
        self.batch_size = batch_size
        self.name = f"whisperx-{model}"

        _allow_pyannote_checkpoint_load()
        self._model = whisperx.load_model(model, device, compute_type=compute_type)
        self._align_model, self._align_meta = whisperx.load_align_model(
            language_code=language, device=device
        )

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        audio = self._whisperx.load_audio(audio_path)
        result = self._model.transcribe(audio, batch_size=self.batch_size)
        result = self._whisperx.align(
            result["segments"],
            self._align_model,
            self._align_meta,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return _extract_word_timings(result)

    def close(self) -> None:  # pragma: no cover - cleanup
        import gc

        try:
            import torch

            del self._model
            del self._align_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
