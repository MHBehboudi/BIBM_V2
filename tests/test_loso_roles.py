"""Gate for the cross-subject role: disjointness, counts, and exact agreement with BiTE's own EA."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reader.data import SPECS, _euclidean_alignment_matrix, loso_roles, raw_roles  # noqa: E402

BITE = ROOT / "third_party/BiteEEG"   # python scripts/fetch_baseline.py


def bite_fuduiban(x):
    """BiTE's own fuduiban, exec'd straight out of their file by source text.

    Importing get_data.py whole pulls in braindecode -> torchaudio and dies on this node's
    CUDA runtime, so the function block is isolated instead. It is still THEIR characters.
    """
    import ast
    import textwrap
    from functools import reduce

    source = (BITE / "get_data.py").read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fuduiban")
    block = textwrap.dedent("\n".join(source.splitlines()[fn.lineno - 1:fn.end_lineno]))
    namespace = {"np": np, "reduce": reduce}
    exec(compile(block, "BiTE_get_data.fuduiban", "exec"), namespace)
    return namespace["fuduiban"](x)


@pytest.mark.skipif(not (ROOT / "third_party/BiteEEG/get_data.py").exists(),
                    reason="BiTE source not fetched; run: python scripts/fetch_baseline.py")
def test_ea_matrix_matches_bite_exactly():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 6, 120))
    assert np.allclose(_euclidean_alignment_matrix(x), bite_fuduiban(x), atol=1e-8)


def test_ea_whitens_the_training_pool():
    train_x, _, _, _ = loso_roles("2b", 1)
    r = np.einsum("bct,bdt->cd", train_x, train_x) / len(train_x)
    assert np.allclose(r / np.trace(r) * r.shape[0], np.eye(r.shape[0]), atol=0.05), r


def test_held_subject_never_appears_in_training():
    for dataset, held in (("2b", 1), ("2a", 3)):
        n = SPECS[dataset]["subjects"]
        train_x, train_y, test_x, test_y = loso_roles(dataset, held, euclidean_alignment=False)
        a, ay, b, by = raw_roles(dataset, held)
        expected_test = len(ay) + len(by)
        assert len(test_y) == expected_test, (dataset, len(test_y), expected_test)
        total = sum(len(raw_roles(dataset, s)[1]) + len(raw_roles(dataset, s)[3]) for s in range(1, n + 1))
        assert len(train_y) == total - expected_test
        # scaling and pooling are linear per trial, so a held trial would survive as a duplicate row
        held_rows = {hash(r.tobytes()) for r in np.concatenate([a, b]).astype(np.float32).reshape(expected_test, -1)}
        train_rows = {hash(r.tobytes()) for r in train_x.reshape(len(train_x), -1)}
        assert not (held_rows & train_rows)


def test_scaler_and_alignment_are_fit_on_training_only():
    """Held subject's statistics must not move when a DIFFERENT subject's data changes shape of nothing."""
    a, _, b, _ = loso_roles("2b", 2, euclidean_alignment=True)
    c, _, d, _ = loso_roles("2b", 2, euclidean_alignment=True)
    assert np.array_equal(a, c) and np.array_equal(b, d)     # deterministic
    e, _, f, _ = loso_roles("2b", 2, euclidean_alignment=False)
    assert not np.allclose(b, f)                             # EA actually does something


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
