"""Gates for reader/error_analysis.py, scripts/make_dashboard.py and the ablation arm-matching rule.

CPU-only, no data, no runs: every test builds its own inputs. What is gated here is the arithmetic and the
three claims that would silently corrupt a paper table if they broke -- that the confusion matrix is
row-normalised over TRUE classes, that the headline accuracy is a subject mean and not a trial-pooled one,
and that two arms which differ only in a metadata field are still treated as a matched pair.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import error_analysis as EA  # noqa: E402


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(spec):
    """{(subject, seed): (logits, labels)} from {(sub, seed): (labels, predictions)}."""
    out = {}
    for key, (labels, preds) in spec.items():
        labels = np.asarray(labels)
        logits = np.full((len(labels), int(max(labels.max(), np.max(preds))) + 1), -3.0)
        logits[np.arange(len(labels)), np.asarray(preds)] = 3.0
        out[key] = (logits, labels)
    return out


# ------------------------------------------------------------------------------------------- confusion
def test_confusion_rows_are_normalised_over_true_classes():
    cells = _cells({(1, 2025): ([0, 0, 0, 1], [0, 0, 1, 1])})
    counts, norm = EA.confusion(cells, 2)
    assert counts.sum() == 4
    np.testing.assert_allclose(norm.sum(1), [1.0, 1.0])
    # true class 0 was predicted 0 twice and 1 once -> recall 2/3, not 2/4 and not 2/2.
    np.testing.assert_allclose(norm[0], [2 / 3, 1 / 3])


def test_confusion_leaves_an_absent_true_class_at_zero_rather_than_dividing_by_zero():
    cells = _cells({(1, 2025): ([0, 0], [0, 1])})
    _, norm = EA.confusion(cells, 3)
    assert np.all(norm[2] == 0) and np.isfinite(norm).all()


def test_per_class_recall_is_the_diagonal_of_the_row_normalised_matrix():
    cells = _cells({(1, 2025): ([0, 0, 1, 1], [0, 1, 1, 1])})
    _, norm = EA.confusion(cells, 2)
    np.testing.assert_allclose(100 * np.diag(norm), [50.0, 100.0])


# -------------------------------------------------------------------------------------------- accuracy
def test_headline_accuracy_is_a_subject_mean_not_a_trial_pooled_one():
    """The distinction is the whole reason both numbers are stored: 2b and HGD subjects differ in size."""
    cells = _cells({(1, 2025): ([0] * 100, [0] * 100),          # big subject, perfect
                    (2, 2025): ([0] * 4, [1, 1, 1, 1])})        # small subject, zero
    assert EA.subject_mean_accuracy(cells) == pytest.approx(50.0)
    correct, trials = 100, 104
    assert 100 * correct / trials == pytest.approx(96.15, abs=0.01)   # what pooling would have said


def test_subject_mean_accuracy_averages_seeds_within_a_subject_first():
    cells = _cells({(1, 2025): ([0, 0], [0, 0]), (1, 2026): ([0, 0], [1, 1]),   # subject 1: 100, 0 -> 50
                    (2, 2025): ([0, 0], [0, 0])})                               # subject 2: 100
    assert EA.subject_mean_accuracy(cells) == pytest.approx(75.0)   # (50 + 100) / 2, not 2/3 of runs


def test_seed_stability_is_the_spread_within_a_subject_not_across_subjects():
    cells = _cells({(1, 2025): ([0, 0], [0, 0]), (1, 2026): ([0, 0], [1, 1]),   # subject 1: 100 and 0
                    (2, 2025): ([0, 0], [0, 0]), (2, 2026): ([0, 0], [0, 0])})  # subject 2: 100 and 100
    s = EA.seed_stability(cells)
    assert s["n_subjects"] == 2
    assert s["mean_range_pp"] == pytest.approx(50.0)             # (100 + 0) / 2
    assert s["max_range_pp"] == pytest.approx(100.0)


# ------------------------------------------------------------------------------------- complementarity
def test_complementarity_partitions_every_trial_exactly_once():
    a = _cells({(1, 2025): ([0, 0, 0, 0], [0, 0, 1, 1])})
    b = _cells({(1, 2025): ([0, 0, 0, 0], [0, 1, 0, 1])})
    r = EA.complementarity(a, b)[0]
    assert r["both_correct"] + r["only_a"] + r["only_b"] + r["both_wrong"] == pytest.approx(100.0)
    assert (r["both_correct"], r["only_a"], r["only_b"], r["both_wrong"]) == (25.0, 25.0, 25.0, 25.0)
    assert r["oracle"] == pytest.approx(75.0)                    # everything except both_wrong


def test_oracle_is_an_upper_bound_on_both_arms_and_on_the_ensemble():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    a = _cells({(1, 2025): (y, np.where(rng.random(200) < .7, y, rng.integers(0, 4, 200)))})
    b = _cells({(1, 2025): (y, np.where(rng.random(200) < .7, y, rng.integers(0, 4, 200)))})
    r = EA.complementarity(a, b)[0]
    assert r["oracle"] >= max(r["a"], r["b"]) and r["oracle"] >= r["ensemble"]


def test_complementarity_refuses_arms_evaluated_on_different_trials():
    a = _cells({(1, 2025): ([0, 1], [0, 1])})
    b = _cells({(1, 2025): ([1, 0], [1, 0])})                    # same shape, different labels
    with pytest.raises(SystemExit):
        EA.complementarity(a, b)


# ------------------------------------------------------------------------------------------ reliability
def test_reliability_bins_recover_a_known_ece_of_zero():
    """A model right exactly as often as it is confident has ECE 0 whatever the bin count."""
    n, conf = 4000, 0.8
    y = np.zeros(n, dtype=int)
    pred = np.where(np.arange(n) < conf * n, 0, 1)
    logits = np.zeros((n, 2))
    logits[np.arange(n), pred] = np.log(conf / (1 - conf))       # softmax max == conf
    _, ece, mean_conf, acc = EA.reliability({(1, 2025): (logits, y)})
    assert mean_conf == pytest.approx(conf, abs=1e-6)
    assert acc == pytest.approx(conf, abs=1e-6)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_reliability_reports_the_gap_when_confidence_and_accuracy_disagree():
    n = 1000
    y = np.zeros(n, dtype=int)
    logits = np.zeros((n, 2))
    logits[:, 0] = 10.0                                          # ~100% confident, and always right
    _, ece, conf, acc = EA.reliability({(1, 2025): (logits, y)})
    assert conf > 0.99 and acc == 1.0 and ece < 0.01
    logits[:, 0], logits[:, 1] = 0.0, 10.0                       # ~100% confident, and always wrong
    _, ece, conf, acc = EA.reliability({(1, 2025): (logits, y)})
    assert acc == 0.0 and ece > 0.99


# --------------------------------------------------------------------- the arms-are-matched rule
def test_ablation_matching_ignores_only_the_protocol_name():
    """`protocol` gained a descriptive `name` after some runs of record were trained; the same protocol
    then serialises two ways. Dropping the name must not let a different protocol through."""
    ablation = _module("reader/ablation.py", "ablation")
    old = {"protocol": {"train_role": "100% official training", "test_role": "official test"}}
    new = {"protocol": {"name": "within", "train_role": "100% official training",
                        "test_role": "official test"}}
    loso = {"protocol": {"name": "loso", "train_role": "all other subjects, both sessions, pooled",
                         "test_role": "held-out subject"}}
    assert ablation._field(old, "protocol") == ablation._field(new, "protocol")
    assert ablation._field(loso, "protocol") != ablation._field(new, "protocol")


def test_ablation_matching_leaves_every_other_field_alone():
    ablation = _module("reader/ablation.py", "ablation")
    for key in ("model", "epochs", "parameters", "input_mode", "window_seconds"):
        assert ablation._field({key: {"name": "x", "v": 1}}, key) == {"name": "x", "v": 1}


# ------------------------------------------------------------------------------------------ dashboard
def test_dashboard_payload_cannot_close_its_own_script_tag():
    dash = _module("scripts/make_dashboard.py", "make_dashboard")
    assert dash.HTML.count("__DATA__") == 1
    payload = {"a": "</script><script>alert(1)</script>", "b": [1, 2.5, None]}
    blob = dash.embed(payload)
    assert "</script" not in blob.lower()
    assert json.loads(blob) == payload                      # the escape round-trips exactly
    page = dash.HTML.replace("__DATA__", blob)
    assert page.count('<script id="payload"') == 1


def test_dashboard_page_is_self_contained():
    """It is committed to a public repo and opened from disk: no network, no build step.

    Checked as resource LOADS, not as the string "http": the SVG namespace is a URI that is an
    identifier, never fetched, and asserting on bare URLs would fail on it for no reason.
    """
    page = (ROOT / "results" / "dashboard.html").read_text()
    for forbidden in ("<script src", "<link rel=\"stylesheet\"", "@import", "fetch(",
                      "XMLHttpRequest", "src=\"http", "href=\"http"):
        assert forbidden not in page, f"dashboard.html loads something external: {forbidden}"
