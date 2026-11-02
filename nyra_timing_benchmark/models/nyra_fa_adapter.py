"""nyra forced aligner adapter (``nyra-forced-aligner`` / ``nyra_align``).

WavLM-large features + triphone GMM-HMM with vocal-event units. Two checkpoints:

* ``nyralabs/nyra_forced_aligner_en``     (``nyra_fa``)
* ``nyralabs/nyra_forced_aligner_en_pro`` (``nyra_fa_pro``, gated; needs an
  ``HF_TOKEN`` with access)

The model expects verbatim transcripts with bracketed fillers/events
(``[UM]``, ``[UH]``, ``[laughter]``), so the benchmark's normalized reference
words are mapped back to that convention before alignment.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..normalize import normalize_word
from .base import TimingModel

DEFAULT_MODEL = "nyralabs/nyra_forced_aligner_en"
PRO_MODEL = "nyralabs/nyra_forced_aligner_en_pro"

_EVENT_TAGS = {
    "um": "[UM]",
    "uh": "[UH]",
    "laughter": "[laughter]",
    "cough": "[cough]",
    "breath": "[breath]",
    "sigh": "[sigh]",
    "noise": "[noise]",
}


def _to_nyra_token(word: str) -> str:
    w = word.strip()
    return _EVENT_TAGS.get(normalize_word(w), w)


class NyraForcedAligner(TimingModel):
    requires_transcript = True

    def __init__(
        self,
        pro: bool = False,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
    ):
        try:
            from nyra_align import Aligner
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "nyra_align could not be imported "
                f"({type(e).__name__}: {e}). `pip install nyra-forced-aligner` "
                "(and `apt install espeak-ng`)."
            ) from e

        model_id = model_id or (PRO_MODEL if pro else DEFAULT_MODEL)
        prefix = "nyra_fa_pro" if model_id == PRO_MODEL else "nyra_fa"
        self.name = f"{prefix}-{model_id.split('/')[-1]}"
        self._aligner = Aligner(model_id, device=device, precision=precision)
        self.num_skipped = 0

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        tokens = [_to_nyra_token(w) for w in (words or []) if normalize_word(w)]
        if not tokens:
            if not transcript:
                raise ValueError("NyraForcedAligner requires reference words or transcript.")
            tokens = transcript.split()
        result = self._aligner.align(audio_path, " ".join(tokens))
        self.num_skipped += len(getattr(result, "skipped", []) or [])
        return [(w.word, float(w.start), float(w.end)) for w in result.words]
