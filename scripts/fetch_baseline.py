#!/usr/bin/env python3
"""Clone the BiTE baseline at the exact commit this work was matched against.

BiteEEG ships no LICENSE, so its code is fetched rather than redistributed here.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reader.models.baseline import BITE_COMMIT, BITE_URL, REPO  # noqa: E402


def main():
    if REPO.exists():
        print(f"already present: {REPO}")
        return
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", BITE_URL, str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", BITE_COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == BITE_COMMIT, f"expected {BITE_COMMIT}, got {head}"
    print(f"BiTE baseline at {BITE_COMMIT[:12]} -> {REPO}")


if __name__ == "__main__":
    main()
