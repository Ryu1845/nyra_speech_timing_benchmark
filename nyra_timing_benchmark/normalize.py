"""Word normalization for timing-benchmark text matching.

Normalization is intentionally lightweight: lowercase and strip
leading/trailing punctuation (including brackets/parens around non-speech tags
and trailing cut-off hyphens) while preserving internal characters such as
apostrophes and compound hyphens. This standardises fillers and cut-offs so
ASR predictions match conversational ground truth (e.g. Buckeye).
"""

from __future__ import annotations

from typing import Optional


_FILLER_CANON = {
    "uhm": "um",
    "ahm": "um",
    "umm": "um",
    "ummm": "um",
    "erm": "um",
    "uhh": "uh",
    "uhhh": "uh",
    "hmmm": "hm",
    "mhmm": "hm",
    "mmhmm": "hm",
    "mm-hm": "hm",
    "hmm": "hm",
    "mmm": "hm",
    "mhm": "hm",
    "mm": "hm",
    "mm-hmm": "hm",
    "uh-huh": "uhhuh",
    "umhum": "uhhuh",
}


#: Fillers are speech and are scored, whether written ``um`` or ``[UM]``.
_FILLERS = frozenset({"um", "uh", "hm", "uhhuh"})
#: Non-speech events. They have no well-defined word boundaries and most
#: systems neither model nor emit them, so they are excluded from scoring on
#: both sides (Buckeye's ``[laughter]`` reaches us as the bare word).
_EVENT_WORDS = frozenset({"laughter"})


def is_scored_word(word: Optional[str]) -> bool:
    """True for tokens that take part in word matching and timing metrics.

    Drops empty/punctuation-only tokens, bracketed or parenthesised event tags
    other than fillers (``[laughter]``, ``(cough)``, ``<noise>``) and the bare
    event words in ``_EVENT_WORDS``.
    """
    norm = normalize_word(word)
    if not norm:
        return False
    if norm in _FILLERS:
        return True
    raw = (word or "").strip()
    if raw[:1] in "[(<{" and raw[-1:] in "])>}":
        return False
    return norm not in _EVENT_WORDS


def normalize_word(word: Optional[str]) -> str:
    """Normalize a word for content matching.

    Lowercases and strips *all* leading/trailing punctuation -- including the
    square brackets and parentheses that ASR models emit around non-speech
    tags. This standardises fillers so a model's ``[UH]``/``(laughter)``
    matches Buckeye's ``uh``/``laughter`` ground truth, and drops cut-off
    markers (a trailing hyphen on a truncated word, e.g. ``reti-`` ->
    ``reti``). Internal characters (apostrophes, compound hyphens like
    ``seventy-seven``) are kept.

    Returns an empty string for ``None`` or whitespace/punctuation-only input,
    which callers use to drop spacer/empty tokens.
    """
    if word is None:
        return ""
    w = word.lower().strip()
    while w and not w[0].isalnum():
        w = w[1:]
    while w and not w[-1].isalnum():
        w = w[:-1]
    return _FILLER_CANON.get(w, w)
