"""Edit-distance word alignment.

Aligns predicted words to ground-truth words using Levenshtein DP and returns the
``(pred_idx, gt_idx)`` index pairs for positions that match exactly (after
normalization). These matched pairs are what timing metrics are computed on, so
that timing accuracy is decoupled from transcription errors.

When timings are supplied, ties between equal-cost edit paths are broken by the
summed boundary distance of the matched pairs (time-mediated alignment). Without
this, a repeated word (``"i i i want"`` vs. a predicted ``"i want"``) is always
matched to the *first* ground-truth occurrence regardless of where the model
placed it, which inflates errors on disfluent speech.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .normalize import normalize_word

Timing = Tuple[Optional[float], Optional[float]]

_DEL, _INS, _SUB = 0, 1, 2


def _time_cost(pt: Timing, gt: Timing) -> float:
    ps, pe = pt
    gs, ge = gt
    if ps is None or pe is None or gs is None or ge is None:
        return 0.0
    return abs(ps - gs) + abs(pe - ge)


def align_words(
    pred_words: Sequence[str],
    gt_words: Sequence[str],
    pred_times: Optional[Sequence[Timing]] = None,
    gt_times: Optional[Sequence[Timing]] = None,
) -> List[Tuple[int, int]]:
    """Align predicted to ground-truth words via edit distance.

    Returns a list of ``(pred_idx, gt_idx)`` pairs for exact (normalized) word
    matches along the minimal edit-distance path. If both ``pred_times`` and
    ``gt_times`` are given, the path is the minimal edit-distance path with the
    smallest total boundary error among its matches.
    """
    p = [normalize_word(w) for w in pred_words]
    g = [normalize_word(w) for w in gt_words]
    n, m = len(p), len(g)

    if n == 0 or m == 0:
        return []

    use_time = (
        pred_times is not None
        and gt_times is not None
        and len(pred_times) == n
        and len(gt_times) == m
    )

    # Lexicographic cost (edits, -matches, time). Equal-edit paths can differ in
    # how many exact matches they keep ("x a" vs "a y": del+match+ins or two
    # subs), so matches are maximised before timing is consulted.
    cost = [[(0, 0, 0.0)] * (m + 1) for _ in range(n + 1)]
    back = [[_SUB] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost[i][0] = (i, 0, 0.0)
        back[i][0] = _DEL
    for j in range(1, m + 1):
        cost[0][j] = (j, 0, 0.0)
        back[0][j] = _INS

    for i in range(1, n + 1):
        pi = p[i - 1]
        row, prev = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            e, nm, t = prev[j]
            best = (e + 1, nm, t)
            move = _DEL
            e, nm, t = row[j - 1]
            cand = (e + 1, nm, t)
            if cand < best:
                best, move = cand, _INS
            e, nm, t = prev[j - 1]
            if pi == g[j - 1]:
                if use_time:
                    t += _time_cost(pred_times[i - 1], gt_times[j - 1])
                cand = (e, nm - 1, t)
            else:
                cand = (e + 1, nm, t)
            if cand < best:
                best, move = cand, _SUB
            row[j] = best
            back[i][j] = move

    i, j = n, m
    matches: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        b = back[i][j]
        if b == _SUB:
            if p[i - 1] == g[j - 1]:
                matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif b == _DEL:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches
