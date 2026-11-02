"""CrisperWhisper word-timing adapter (free-decode ASR).

This wraps the local ``crisperwhisper`` package to *transcribe* the audio
verbatim and read per-word start/end times off the decoder cross-attention.
It can run either the ``transformers`` or ``ct2`` backend. It is an **ASR
model** (``requires_transcript = False``): the model predicts its own words and
timings, and the benchmark then aligns those words to the ground-truth word list
(edit-distance matching with filler/cut-off normalization) before scoring
timing on the matched pairs.

Why free decode instead of teacher forcing the reference transcript?
Teacher forcing is only safe when the ground-truth transcript *conforms* to
what the model actually utters. That holds for clean read speech (TIMIT ->
~36ms MAE) but breaks on conversational corpora like Buckeye, whose
``transcript_form`` omits/normalizes words the model (and the audio) clearly
contain -- e.g. GT ``"... my dinner so ..."`` vs audio ``"... my retirement
dinner. So ..."``. Forcing the non-conforming sequence onto the audio
compresses the alignment and produces catastrophic end errors (hundreds of ms)
plus a large negative end-bias around the frequent pauses. Predicting regularly
times the model's *own* (correct) tokens; matching then keeps only the words
that actually correspond, decoupling timing accuracy from transcript mismatch.

Pipeline:
1. Build the verbatim decoder prompt (``[verbatim_1..5]`` + sot/lang/transcribe/
   notimestamps).
2. Free-decode with repair (``generate_with_repair_and_attention``) to get the
   token ids and the head-averaged cross-attention over the model's
   ``generation_config.alignment_heads``.
3. Run the package's Viterbi word alignment over the *full* encoder window
   (mel-energy blank, no audio clipping) to get per-word start/end times. The
   raw mel-energy blank is reshaped (``blank_gamma`` / ``blank_penalty``) so it
   only fires on genuinely silent frames -- the raw blank rates ordinary
   low-energy conversational speech as half-silent and the Viterbi steals word
   edges, collapsing Buckeye words to ~26% of their true length. The reshaped
   blank recovers ~78% (Buckeye 87->42ms MAE, TIMIT unchanged at ~34ms).

Env overrides: ``CRISPERWHISPER_REPO`` (package checkout),
``CRISPERWHISPER_MODEL`` (model path), ``CRISPERWHISPER_BACKEND``
(``transformers`` or ``ct2``), and ``CRISPERWHISPER_CACHE`` (CT2 conversion
cache).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

from .base import TimingModel

DEFAULT_REPO = "/home/ubuntu/code/research"
DEFAULT_MODEL = (
    "/home/ubuntu/data/models/large-v2_verbatimize_longform_with_timing_release"
)
DEFAULT_BACKEND = "transformers"


class CrisperWhisperModel(TimingModel):
    requires_transcript = False

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        sharpen: float = 5.0,
        blank_source: str = "mel",
        blank_gamma: float = 3.0,
        blank_penalty: float = 3.0,
        split_gap_max_s: float = 0.1,
        repo_path: Optional[str] = None,
        backend: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        repo = repo_path or os.environ.get("CRISPERWHISPER_REPO", DEFAULT_REPO)
        if repo and repo not in sys.path and os.path.isdir(repo):
            sys.path.insert(0, repo)
        try:
            from crisperwhisper.prompt import PromptBuilder
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "Could not import the 'crisperwhisper' package "
                f"({type(e).__name__}: {e}). Set CRISPERWHISPER_REPO to its "
                "checkout and ensure torch + transformers are installed."
            ) from e

        self.model_path = model_path or os.environ.get(
            "CRISPERWHISPER_MODEL", DEFAULT_MODEL
        )
        self.backend = (
            backend
            or os.environ.get("CRISPERWHISPER_BACKEND", DEFAULT_BACKEND)
        ).lower()
        if self.backend not in {"transformers", "ct2"}:
            raise ValueError(
                "CrisperWhisper timing backend must be "
                "'transformers' or 'ct2', "
                f"got {self.backend!r}."
            )
        self.language = language
        self.sharpen = sharpen
        self.blank_source = blank_source
        self.blank_gamma = blank_gamma
        self.blank_penalty = blank_penalty
        self.split_gap_max_s = split_gap_max_s
        self.name = (
            "crisperwhisper-"
            + self.backend
            + "-"
            + os.path.basename(self.model_path.rstrip("/"))
        )

        if self.backend == "ct2":
            try:
                from crisperwhisper.converter import ensure_ct2_model
                from crisperwhisper.engine import CT2Engine
            except Exception as e:  # pragma: no cover - optional dep
                raise ImportError(
                    "The CrisperWhisper CT2 timing backend requires "
                    "crisperwhisper[ct2]."
                ) from e
            ct2_path = ensure_ct2_model(
                self.model_path,
                quantization=compute_type,
                cache_dir=cache_dir,
            )
            self._engine = CT2Engine(
                ct2_path, device=device, compute_type=compute_type
            )
        else:
            try:
                from crisperwhisper.transformers_engine import TransformersEngine
            except Exception as e:  # pragma: no cover - optional dep
                raise ImportError(
                    "The CrisperWhisper transformers timing backend requires "
                    "crisperwhisper[transformers]."
                ) from e
            self._engine = TransformersEngine(
                self.model_path, device=device, compute_type=compute_type
            )
        # Load generation_config.alignment_heads from the selected backend.
        self._engine.enable_attention()
        self._prompt = PromptBuilder(self._engine, language=language).verbatim()

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        import numpy as np
        import soundfile as sf
        import torch

        from crisperwhisper.word_timing import extract_word_timings

        audio, sr = sf.read(audio_path)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        feats, mel = self._engine.extract_features_with_mel(audio)

        # Free verbatim decode (with hallucination repair) + recover the
        # 1-to-1 head-averaged cross-attention in one teacher-forced pass over
        # the *generated* tokens.
        gen_ids, attention = self._engine.generate_with_repair_and_attention(
            feats, self._prompt, max_length=256
        )

        del feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not gen_ids:
            return []

        # Full-window Viterbi word alignment (mel blank, no audio clipping --
        # the silence padding is absorbed by the blank states). ``blank_gamma``
        # / ``blank_penalty`` reshape the mel blank so it only fires on
        # genuinely silent frames, countering the word-collapse on
        # conversational speech (Buckeye 87->46ms MAE, TIMIT unchanged).
        # ``split_gap_max_s`` then shares short inter-word silences (the gaps the
        # Viterbi parks in blank states) evenly between neighbouring words so
        # tight boundaries become contiguous, while real pauses are left intact.
        word_ts = extract_word_timings(
            self._engine,
            gen_ids,
            attention,
            mel,
            sharpen=self.sharpen,
            blank_source=self.blank_source,
            blank_gamma=self.blank_gamma,
            blank_penalty=self.blank_penalty,
            split_gap_max_s=self.split_gap_max_s,
        )

        out: List[Tuple[str, float, float]] = []
        for wt in word_ts:
            if (
                wt.word
                and wt.word.strip()
                and wt.start is not None
                and wt.end is not None
            ):
                out.append((wt.word.strip(), float(wt.start), float(wt.end)))
        return out

    def close(self) -> None:  # pragma: no cover - cleanup
        import gc

        try:
            import torch

            del self._engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
