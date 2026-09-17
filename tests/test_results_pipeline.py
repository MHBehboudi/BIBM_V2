"""Gates for the results pipeline: the rules that decide WHICH runs a paper number comes from, and the
metrics computed on top of them. Every test is CPU-only and needs no data."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------- metrics
def test_cohen_kappa_matches_the_definition():
    export = _script("export_runs")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    p = np.where(rng.random(500) < .7, y, rng.integers(0, 4, 500))
    confusion = np.zeros((4, 4))
    np.add.at(confusion, (y, p), 1)
    po = np.trace(confusion) / 500
    pe = (confusion.sum(0) * confusion.sum(1)).sum() / 500 ** 2
    assert export.cohen_kappa(y, p, 4) == pytest.approx((po - pe) / (1 - pe))
    assert export.cohen_kappa(y, y, 4) == pytest.approx(1.0)


@pytest.mark.parametrize("dataset,deadline,expected_index", [
    ("2a", 1.0, 7), ("2a", 3.0, 23), ("2a", 4.0, 31),            # 128 ms tokens: ceil(3000/128)=24 -> index 23
    ("sdssvep", 0.25, 15), ("sdssvep", 0.5, 31), ("sdssvep", 1.0, 63),   # 15.625 ms tokens
])
def test_deadline_token_ends_at_or_after_the_deadline(dataset, deadline, expected_index):
    from reader.anytime import deadline_index
    from reader.data import SPECS
    spec = SPECS[dataset]
    pool_ms = 1000 * spec["pool"] / spec["fs"]
    n_tokens = -(-spec["samples"] // spec["pool"])
    i = deadline_index(deadline, pool_ms, n_tokens)
    assert i == expected_index
    assert (i + 1) * pool_ms >= deadline * 1000 - 1e-6


# ------------------------------------------------------------------------------------- record rules
def _run(root, rel, device, acc=.8):
    d = root / rel
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"status": "complete", "device": device, "final_test": {"acc": acc}}))
    return d


def test_record_prefers_full_gpu_rerun_and_supersedes_clip5_bite_loso(tmp_path, monkeypatch):
    link = _script("link_record_runs")
    h = tmp_path / "harness" / "runs"
    _run(h, "cohort_20260915/bite/2a_S1_seed2025", "NVIDIA H100 NVL MIG 3g.47gb", .70)
    _run(h, "fullgpu_20260917/cohort_20260915/bite/2a_S1_seed2025", "NVIDIA H200 NVL", .72)
    _run(h, "cohort_20260915/bite/2a_S2_seed2025", "NVIDIA H100 NVL MIG 3g.47gb")        # no re-run yet
    _run(h, "cohort_20260915/reader/2a_S1_seed2025", "NVIDIA H100 80GB HBM3")
    _run(h, "loso_20260915/bite/2a_S1_seed2025", "NVIDIA H200 NVL")                       # clip 5: superseded
    _run(h, "loso_20260917/bite/2a_S1_seed2025", "NVIDIA H200 NVL")                       # clip 0: record
    runs = tmp_path / "repo_runs"
    monkeypatch.setattr(link, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["link", "--harness", str(h), "--runs", str(runs)])
    link.main()
    rec = {(r["view"], r["cell"]): r for r in json.loads((runs / "RECORD.json").read_text())["runs"]}
    assert (runs / "cohort/bite/2a_S1_seed2025").resolve() == (h / "fullgpu_20260917/cohort_20260915/bite/2a_S1_seed2025")
    assert rec[("cohort/bite", "2a_S1_seed2025")]["status"].startswith("full-GPU re-run")
    assert (runs / "mig/cohort/bite/2a_S1_seed2025").resolve() == (h / "cohort_20260915/bite/2a_S1_seed2025")
    assert rec[("cohort/bite", "2a_S2_seed2025")]["status"] == "mig_pending"
    assert (runs / "loso/bite/2a_S1_seed2025").resolve() == (h / "loso_20260917/bite/2a_S1_seed2025")
    assert (runs / "superseded/loso_bite_clip5/2a_S1_seed2025").resolve() == (h / "loso_20260915/bite/2a_S1_seed2025")


def test_exact_duration_records_of_replaced_runs_are_dropped(tmp_path, monkeypatch):
    collect = _script("collect_exact_duration")
    runs = tmp_path / "runs"
    exact = runs / "exact_duration" / "2a" / "bite"
    exact.mkdir(parents=True)
    base = {"dataset": "2a", "subject": 1, "arm": "bite", "acc": {"1000": .7}, "gate": {},
            "full_length_matches_logged": True}
    for seed, device in ((2025, "NVIDIA H100 NVL MIG 3g.47gb"), (2026, "NVIDIA H200 NVL")):
        (exact / f"2a_S1_seed{seed}.json").write_text(json.dumps(
            {**base, "seed": seed, "run": f"2a_S1_seed{seed}", "device_trained": device, "final_test_acc_logged": .7}))
    _run(runs, "cohort/bite/2a_S1_seed2025", "NVIDIA H200 NVL", .7)      # replaced: GPU changed
    _run(runs, "cohort/bite/2a_S1_seed2026", "NVIDIA H200 NVL", .7)      # unchanged
    out = tmp_path / "exact.json"
    monkeypatch.setattr(sys, "argv", ["collect", "--exact", str(runs / "exact_duration"), "--runs", str(runs),
                                      "--out", str(out)])
    monkeypatch.setattr(collect, "ROOT", tmp_path)
    collect.main()
    kept = json.loads(out.read_text())["records"]
    assert [r["seed"] for r in kept] == [2026]


# ------------------------------------------------------------------------------------------ efficiency
@pytest.mark.parametrize("dataset", ["2a", "2b", "hgd", "sdssvep"])
def test_single_decision_path_equals_the_model(dataset):
    """The latency table times ONE decision; it must be the decision the model makes."""
    efficiency = _script("efficiency")
    from reader.data import SPECS
    from reader.models import registry
    torch.manual_seed(0)
    model, *_ = registry.build("reader", dataset)
    for block in list(model.tcn.blocks) + list(model.backward_tcn.blocks):   # leave the zero-init identity
        torch.nn.init.normal_(block.conv2.weight, std=.2)
    with torch.no_grad():
        model.backward_gate.normal_()
    model.eval()
    spec = SPECS[dataset]
    for samples in (spec["samples"] // 3, spec["samples"]):
        x = torch.randn(2, spec["channels"], samples)
        with torch.no_grad():
            ref = model(x)["logits"]
            assert torch.allclose(efficiency.reader_decision(model, x), ref, atol=1e-5)


# ------------------------------------------------------------------------------------ reproduce scripts
@pytest.mark.parametrize("script", ["reproduce_within.sh", "reproduce_loso.sh", "reproduce_anytime.sh"])
def test_bite_is_trained_without_gradient_clipping(script):
    """BiTE's release does not clip; every reported BiTE run uses --clip 0."""
    text = (ROOT / "scripts" / script).read_text()
    bite_arms = re.findall(r"--arm [\"']?[\w.${}]*:bite:([^\s\"']*)", text)
    add_bank = "add_bank()" in text and '[ "$1" = bite ] && extra=",--clip,0"' in text
    assert bite_arms or add_bank
    for extra in bite_arms:
        assert "--clip,0" in extra, f"{script}: bite arm without --clip 0: {extra}"
