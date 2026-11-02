"""CTC-segmentation forced-aligner adapter.

Aligns reference words to audio using wav2vec2 (``WAV2VEC2_ASR_BASE_960H``) CTC
emissions and the ``ctc-segmentation`` algorithm. Requires ``torchaudio`` and the
``ctc_segmentation`` package (the latter must be built against the active numpy
ABI).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .base import TimingModel


def _normalize(word: str) -> str:
    w = (word or "").strip().lower()
    w = w.rstrip(".,;:!?\"'")
    w = w.lstrip("\"'")
    return w


def _sanitize_for_ctc(word: str) -> str:
    w = _normalize(word)
    w = w.rstrip("=")
    return w.replace("_", "").replace("'", "").replace("-", "")


class CtcSegmentationModel(TimingModel):
    requires_transcript = True

    def __init__(self, device: str = "cuda"):
        try:
            import torch
            import torchaudio
            from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H as bundle
            from ctc_segmentation import (
                CtcSegmentationParameters,
                ctc_segmentation,
                determine_utterance_segments,
                prepare_text,
            )
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "torchaudio and ctc_segmentation are required for the CTC-"
                "segmentation adapter. Install with `pip install ctc-segmentation` "
                "in an environment where it is built against the active numpy."
            ) from e

        self._torch = torch
        self._torchaudio = torchaudio
        self._bundle = bundle
        self._ctc_segmentation = ctc_segmentation
        self._determine_utterance_segments = determine_utterance_segments
        self._prepare_text = prepare_text

        self.name = "ctc_segmentation-wav2vec2_base_960h"
        self.device = device if torch.cuda.is_available() else "cpu"
        self._labels = list(bundle.get_labels())
        self._config = CtcSegmentationParameters()
        self._config.char_list = self._labels
        self._config.blank = self._labels.index("-")
        self._config.space = "|"
        self._config.update_excluded_characters()
        self._model = bundle.get_model().to(self.device).eval()

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        torch = self._torch
        ta = self._torchaudio

        if words:
            words_gt = [_normalize(w) for w in words if _normalize(w)]
        elif transcript:
            words_gt = [_normalize(w) for w in transcript.split() if _normalize(w)]
        else:
            raise ValueError("CtcSegmentationModel requires reference words or transcript.")

        wav, sr = ta.load(audio_path)
        if sr != self._bundle.sample_rate:
            wav = ta.functional.resample(wav, sr, self._bundle.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(self.device)

        with torch.inference_mode():
            emissions, _ = self._model(wav)
            log_probs = torch.log_softmax(emissions, dim=-1)[0].cpu().numpy()

        duration_s = wav.shape[1] / self._bundle.sample_rate
        self._config.index_duration = duration_s / log_probs.shape[0]

        words_in = [_sanitize_for_ctc(w).upper() for w in words_gt]
        gt_mat, utt_begin_indices = self._prepare_text(
            self._config, words_in, char_list=self._labels
        )
        timings, char_probs, _ = self._ctc_segmentation(
            self._config, log_probs, gt_mat
        )
        segments = self._determine_utterance_segments(
            self._config, utt_begin_indices, char_probs, timings, words_in
        )
        return [
            (w, float(start), float(end))
            for w, (start, end, _conf) in zip(words_gt, segments)
        ]
