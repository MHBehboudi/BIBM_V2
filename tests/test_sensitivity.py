"""Gates for reader/sensitivity.py, the hyper-parameter sensitivity scorer.

CPU-only and data-free: every test builds its own cells. What is gated is the arithmetic that the
supplementary tables are made of (the nAUC quadrature, its integration range, the paired difference
and its bootstrap) and the four guards that stop a mislabelled cell from becoming a result -- a
record computed from a different run, a pooling factor that disagrees with the plan, a
prefix-supervision weight that disagrees with the plan, and accuracy read in the wrong units.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import sensitivity as S  # noqa: E402
from reader.data import SPECS  # noqa: E402

GRID_2A = [125, 250, 375, 500, 625, 750, 875, 1000]
GRID_SD = [32, 64, 96, 128, 160, 192, 224, 256]


# ------------------------------------------------------------------------------------- quadrature
def test_nauc_of_a_constant_curve_is_that_constant():
    """Normalisation by the duration range is what makes nAUC comparable with an accuracy."""
    assert S.nauc(GRID_2A, [70.0] * 8, 250.0) == pytest.approx(70.0)


def test_nauc_of_a_straight_line_is_its_midpoint():
    acc = [10.0 * i for i in range(8)]                     # 0 .. 70 over 0.5 .. 4.0 s
    assert S.nauc(GRID_2A, acc, 250.0) == pytest.approx(35.0)


def test_paper_range_drops_the_first_grid_point_only():
    """The manuscript's nAUC integrates from 1.0 s on 2a/2b and 0.25 s on SD-SSVEP."""
    acc = [0.0] + [100.0] * 7
    full = S.nauc(GRID_2A, acc, 250.0)
    paper = S.nauc(GRID_2A, acc, 250.0, start=S.PAPER_START["2a"])
    assert paper == pytest.approx(100.0)
    assert full < paper                                    # the discarded point is the weak one
    assert S.PAPER_START["2a"] / 250.0 == pytest.approx(1.0)
    assert S.PAPER_START["sdssvep"] / SPECS["sdssvep"]["fs"] == pytest.approx(0.25)


def test_trapezoid_weights_interior_points_twice():
    """A spike at an interior duration must count more than the same spike at an endpoint."""
    interior, edge = [0.0] * 8, [0.0] * 8
    interior[4] = 100.0
    edge[-1] = 100.0
    assert S.nauc(GRID_2A, interior, 250.0) == pytest.approx(2 * S.nauc(GRID_2A, edge, 250.0))


# ------------------------------------------------------------------------------------- bootstrap
def test_bootstrap_is_centred_on_the_paired_mean_and_is_deterministic():
    d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    mean, lo, hi = S.bootstrap_ci(d, seed=0)
    assert mean == pytest.approx(5.0)
    assert lo < mean < hi
    assert S.bootstrap_ci(d, seed=0) == S.bootstrap_ci(d, seed=0)


def test_bootstrap_of_an_all_positive_difference_excludes_zero():
    mean, lo, hi = S.bootstrap_ci([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], seed=0)
    assert lo > 0


# ------------------------------------------------------------------------------ cells and pairing
def _cell(dataset, arm, lam, pool, per_subject, subjects=(1, 2, 3), seeds=(2025, 2026, 2027)):
    cell = S.Cell(dataset, arm, lam, pool, "synthetic")
    grid = GRID_2A if dataset != "sdssvep" else GRID_SD
    for sub in subjects:
        for seed in seeds:
            value = per_subject[sub]
            cell.add_run({"subject": sub, "seed": seed, "final_test": {"acc": value / 100},
                          "device": "test", "parameters": 1},
                         {str(n): value / 100 for n in grid})
    return cell


def test_accuracy_is_reported_in_points_not_fractions():
    """exact_duration stores fractions; a missing x100 would turn +9.31 points into +0.09."""
    cell = _cell("2a", "reader", 0.0, 32, {1: 70.0, 2: 80.0, 3: 90.0})
    assert cell.nauc_by_subject(full_range=False)[1] == pytest.approx(70.0)
    assert cell.endpoint_by_subject()[1] == pytest.approx(70.0)


def test_paired_difference_uses_subjects_as_units_not_runs():
    reader = _cell("2a", "reader", 0.0, 32, {1: 80.0, 2: 80.0, 3: 80.0})
    control = _cell("2a", "compact", 0.0, 32, {1: 70.0, 2: 75.0, 3: 79.0})
    row = S.compare(reader, control)
    assert row["subjects"] == 3 and row["runs"] == [9, 9]
    assert row["delta_nauc"] == pytest.approx((10 + 5 + 1) / 3)
    assert row["wins"] == 3 and row["losses"] == 0


def test_comparison_uses_only_subjects_present_in_both_arms():
    reader = _cell("2a", "reader", 0.0, 32, {1: 80.0, 2: 80.0, 3: 80.0}, subjects=(1, 2, 3))
    control = _cell("2a", "compact", 0.0, 32, {1: 70.0, 2: 70.0}, subjects=(1, 2))
    assert S.compare(reader, control)["subjects"] == 2


def test_token_count_follows_the_pooling_factor():
    one = {1: 70.0}
    assert _cell("2a", "reader", 0.0, 64, one, subjects=(1,)).token_count() == 16
    assert _cell("2a", "reader", 0.0, 16, one, subjects=(1,)).token_count() == 63
    assert _cell("sdssvep", "reader", 0.0, 4, one, subjects=(1,)).token_count() == 64


# ----------------------------------------------------------------------------------------- guards
def _record(tmp_path, dataset="2a", pool=32, lam=0.3, acc=0.8, logged=None):
    """A one-cell (run dir, exact dir) pair on disk, as `load` expects to find them."""
    run = tmp_path / "runs" / f"{dataset}_S1_seed2025"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(
        {"subject": 1, "seed": 2025, "prefix_weight": lam, "parameters": 1, "device": "test",
         "final_test": {"acc": acc}}))
    exact = tmp_path / "exact"
    exact.mkdir(exist_ok=True)
    (exact / f"{dataset}_S1_seed2025.json").write_text(json.dumps(
        {"pool": pool, "final_test_acc_logged": acc if logged is None else logged,
         "acc": {str(n): acc for n in GRID_2A}}))
    return tmp_path / "runs", exact


def test_load_accepts_a_matched_cell(tmp_path):
    runs, exact = _record(tmp_path)
    cells = S.load(None, None, [("2a", "reader", 0.3, 32, runs, exact)])
    assert cells[("2a", "reader", 0.3, 32)].n_runs == 1


def test_load_rejects_a_record_computed_from_a_different_run(tmp_path):
    runs, exact = _record(tmp_path, acc=0.8, logged=0.7)
    with pytest.raises(SystemExit, match="different run"):
        S.load(None, None, [("2a", "reader", 0.3, 32, runs, exact)])


def test_load_rejects_a_pooling_factor_that_disagrees_with_the_plan(tmp_path):
    runs, exact = _record(tmp_path, pool=16)
    with pytest.raises(SystemExit, match="pool"):
        S.load(None, None, [("2a", "reader", 0.3, 32, runs, exact)])


def test_load_rejects_a_prefix_weight_that_disagrees_with_the_plan(tmp_path):
    runs, exact = _record(tmp_path, lam=1.0)
    with pytest.raises(SystemExit, match="lambda"):
        S.load(None, None, [("2a", "reader", 0.3, 32, runs, exact)])


def test_missing_cells_are_skipped_not_invented(tmp_path):
    assert S.load(None, None, [("2a", "reader", 0.3, 32, tmp_path / "nope", tmp_path / "nope")]) == {}


# --------------------------------------------------------------------- the record's own anchor row
@pytest.mark.skipif(not (ROOT / "results/sensitivity_plan.json").exists(),
                    reason="needs the run record")
def test_anchor_rows_reproduce_the_published_nauc_differences():
    """lambda 0.3 at the default pooling factor is the manuscript's REACT+PS - no-reverse+PS row."""
    plan = json.loads((ROOT / "results/sensitivity_plan.json").read_text())
    cells = S.load(None, None, [(c["dataset"], c["arm"], c["lambda"], c["pool"], c["runs"], c["exact"])
                                for c in plan])
    expected = {"2a": 9.31, "2b": 1.13, "sdssvep": 12.30}
    for ds, want in expected.items():
        key_r, key_c = (ds, "reader", 0.3, SPECS[ds]["pool"]), (ds, "compact", 0.3, SPECS[ds]["pool"])
        if key_r not in cells or key_c not in cells:
            pytest.skip(f"{ds} prefix-supervised anchor cells are not in this checkout")
        assert S.compare(cells[key_r], cells[key_c])["delta_nauc"] == pytest.approx(want, abs=0.01)
