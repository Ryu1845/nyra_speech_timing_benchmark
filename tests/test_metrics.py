import numpy as np
import pytest

from nyra_timing_benchmark.align import align_words
from nyra_timing_benchmark.evaluate import evaluate_dataset, evaluate_sample
from nyra_timing_benchmark.metrics import iou


class _GT:
    def __init__(self, words, timings):
        self.words = words
        self.timings = timings


def test_align_without_times_is_unchanged():
    assert align_words(["i", "want"], ["i", "i", "i", "want"]) == [(0, 0), (1, 3)]


def test_align_uses_time_to_pick_repeated_word():
    gt_w = ["i", "i", "i", "want"]
    gt_t = [(0.0, 0.1), (0.3, 0.4), (0.8, 0.9), (1.0, 1.4)]
    pred_w = ["i", "want"]
    pred_t = [(0.82, 0.9), (1.0, 1.4)]
    assert align_words(pred_w, gt_w, pred_t, gt_t) == [(0, 2), (1, 3)]


def test_align_keeps_matches_over_cheaper_substitutions():
    # "x a" vs "a y": two subs tie del+match+ins on edit count; the match must win.
    al = align_words(["x", "a"], ["a", "y"], [(0, 1), (1, 2)], [(5, 6), (7, 8)])
    assert al == [(1, 0)]


def test_iou_basic():
    assert iou(0, 1, 0, 1) == 1.0
    assert iou(0, 1, 1, 2) == 0.0
    assert iou(0, 2, 1, 3) == pytest.approx(1 / 3)


def test_each_word_contributes_two_boundaries_once():
    gt = {"a": _GT(["x", "y"], [(0.0, 1.0), (1.0, 2.0)])}
    pred = {"a": (["x", "y"], [(0.1, 1.0), (1.0, 2.3)])}
    o = evaluate_dataset(gt, pred, collars=[0.05]).overall
    assert o["num_scored_words"] == 2
    # per-word errors: (0.1+0)/2, (0+0.3)/2 -> mean 0.1, median 0.1
    assert o["mae_ms"]["overall"] == pytest.approx(100.0)
    assert o["median_ae_ms"]["overall"] == pytest.approx(100.0)
    assert o["median_ae_ms"]["start"] == pytest.approx(50.0)
    assert o["median_ae_ms"]["end"] == pytest.approx(150.0)
    ious = [iou(0.1, 1.0, 0.0, 1.0), iou(1.0, 2.3, 1.0, 2.0)]
    assert o["miou"] == pytest.approx(np.mean(ious))
    assert o["median_iou"] == pytest.approx(np.median(ious))


def test_word_recall_counts_unmatched_gt_words():
    gt = {"a": _GT(["x", "y", "z", "w"], [(0, 1), (1, 2), (2, 3), (3, 4)])}
    pred = {"a": (["x", "q"], [(0, 1), (1, 2)])}
    o = evaluate_dataset(gt, pred, collars=[0.05]).overall
    assert o["word_recall"] == pytest.approx(0.25)


def test_missing_sample_scored_empty():
    gt = {"a": _GT(["x"], [(0, 1)]), "b": _GT(["y"], [(0, 1)])}
    pred = {"a": (["x"], [(0, 1)])}
    o = evaluate_dataset(gt, pred, collars=[0.05]).overall
    assert o["coverage"] == 0.5
    assert o["word_recall"] == 0.5
    assert o["f1_standard"]["50ms"]["recall"] == 0.5


def test_events_excluded_and_f1_uses_alignment():
    gt = {"a": _GT(["x", "laughter", "y"], [(0, 1), (1, 2), (2, 3)])}
    pred = {"a": (["[laughter]", "x", "y", "x"], [(0, 0.5), (0, 1), (2, 3), (2, 3)])}
    o = evaluate_dataset(gt, pred, collars=[0.05]).overall
    assert o["num_gt_words"] == 2 and o["num_gt_excluded_events"] == 1
    assert o["word_recall"] == 1.0
    # the extra "x" is an insertion: it may not steal a hit from a GT word elsewhere
    f = o["f1_standard"]["50ms"]
    assert (f["tp"], f["fp"], f["fn"]) == (2, 1, 0)


def test_sample_medians():
    sr = evaluate_sample(["x"], [(0.1, 1.2)], ["x"], [(0.0, 1.0)], collars=[0.05])
    assert sr.median_ae_ms == pytest.approx(150.0)
    sr = evaluate_sample(["x", "y"], [(0.1, 1.0), (1.0, 2.3)], ["x", "y"], [(0, 1), (1, 2)], collars=[0.05])
    assert sr.median_ae_ms == pytest.approx(100.0)


def test_verbatim_filler_spellings_match_reference():
    from nyra_timing_benchmark.normalize import is_scored_word, normalize_word

    for w, canon in [("Um,", "um"), ("uh...", "uh"), ("Umm", "um"), ("Uhh,", "uh"),
                     ("Mm-hmm.", "hm"), ("Uh-huh.", "uhhuh"), ("[UM]", "um"), ("r-", "r")]:
        assert normalize_word(w) == canon
        assert is_scored_word(w)
