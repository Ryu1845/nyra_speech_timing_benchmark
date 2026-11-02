"""NeMo forced-aligner adapter.

Uses a CTC ASR model (default ``stt_en_conformer_ctc_large``) plus NeMo's Viterbi
forced-alignment utilities to align the reference words to audio. Requires the
``nemo_toolkit[asr]`` package.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .base import TimingModel


def _normalize(word: str) -> str:
    w = (word or "").strip().lower()
    w = w.rstrip(".,;:!?\"'")
    w = w.lstrip("\"'")
    return w


class NemoForcedAligner(TimingModel):
    requires_transcript = True

    def __init__(
        self,
        pretrained_name: str = "stt_en_conformer_ctc_large",
        device: str = "cuda",
    ):
        try:
            import torch
            from nemo.collections.asr.models import ASRModel
            from nemo.collections.asr.parts.utils.aligner_utils import (  # noqa: F401
                Segment,
                Word,
                add_t_start_end_to_utt_obj,
                get_batch_variables,
                viterbi_decoding,
            )
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "NeMo is required for the NeMo forced aligner. "
                "Install with `pip install nemo_toolkit[asr]`."
            ) from e

        self._torch = torch
        self._aligner = __import__(
            "nemo.collections.asr.parts.utils.aligner_utils",
            fromlist=["*"],
        )
        self.name = f"nemo_fa-{pretrained_name}"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = ASRModel.from_pretrained(pretrained_name)
        self._model.eval().to(self.device)
        self._timestep_duration: Optional[float] = None

    def _extract_words(self, utt_obj) -> List[Tuple[str, float, float]]:
        aligner = self._aligner
        out: List[Tuple[str, float, float]] = []

        def add_word(word) -> None:
            if word.t_start is None or word.t_end is None or word.t_start < 0:
                return
            text = _normalize(word.text or "")
            if text:
                out.append((text, float(word.t_start), float(word.t_end)))

        for seg_or_tok in utt_obj.segments_and_tokens:
            if isinstance(seg_or_tok, aligner.Segment):
                for word_or_tok in seg_or_tok.words_and_tokens:
                    if isinstance(word_or_tok, aligner.Word):
                        add_word(word_or_tok)
            elif isinstance(seg_or_tok, aligner.Word):
                add_word(seg_or_tok)
        return out

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        aligner = self._aligner
        if words:
            text = " ".join(_normalize(w) for w in words if _normalize(w))
        elif transcript:
            text = " ".join(_normalize(w) for w in transcript.split() if _normalize(w))
        else:
            raise ValueError("NemoForcedAligner requires reference words or transcript.")

        (
            log_probs_batch,
            y_batch,
            t_batch,
            u_batch,
            utt_obj_batch,
            self._timestep_duration,
        ) = aligner.get_batch_variables(
            audio=[audio_path],
            model=self._model,
            segment_separators=[],
            align_using_pred_text=False,
            gt_text_batch=[text],
            output_timestep_duration=self._timestep_duration,
        )

        alignments = aligner.viterbi_decoding(
            log_probs_batch, y_batch, t_batch, u_batch, self.device
        )
        utt_obj = aligner.add_t_start_end_to_utt_obj(
            utt_obj_batch[0], alignments[0], self._timestep_duration
        )
        return self._extract_words(utt_obj)

    def close(self) -> None:  # pragma: no cover - cleanup
        import gc

        try:
            del self._model
            gc.collect()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass
