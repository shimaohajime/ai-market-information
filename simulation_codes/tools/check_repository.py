#!/usr/bin/env python3
"""Check the public simulation package for required source and accidental secrets.

The check deliberately examines only Git-tracked files. Local licensed inputs,
private notes, and ignored build artifacts may coexist beside this package and
must not make a clean public release fail its structural preflight.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "CITATION.cff",
    "docs/DATA_AVAILABILITY.md",
    "docs/REPRODUCIBILITY_STATUS.md",
    "results/aggregate/README.md",
    "simulations/population/gaussian/benchmark.py",
    "simulations/population/gaussian/tests/test_population.py",
    "simulations/population/eight_state/audit_theorem2.py",
    "simulations/neural/first_deployment/run_phase2.py",
    "simulations/neural/e3_e4_information_value/run_measurement.py",
    "simulations/neural/e5_retraining/run.py",
    "simulations/neural/e6_e7_group_retraining/run_e6.py",
    "simulations/neural/e9_e10_extensions/run_e9.py",
    "tools/si_figures/README.md",
]
PROHIBITED_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".parquet",
    ".feather",
    ".sqlite",
    ".pkl",
    ".pickle",
    ".ttf",
    ".otf",
}
TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".tex",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".cff",
    ".cfg",
    ".ini",
    ".sh",
}
LEAK_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|PRIVATE) PRIVATE KEY-----"),
    "api-key": re.compile(r"(?<![a-zA-Z0-9])sk-[a-zA-Z0-9]{20,}"),
    "homepath": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "windows-userpath": re.compile(r"C:\\Users\\[^\\]+\\", re.IGNORECASE),
    "one-drive": re.compile(r"OneDrive-(?:The)?Pennsylvania", re.IGNORECASE),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [ROOT / value for value in result.stdout.decode().split("\0") if value]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        print("MISSING", missing)
        raise SystemExit(2)

    issues: list[tuple[str, str]] = []
    files = tracked_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in PROHIBITED_EXTENSIONS:
            issues.append((str(relative), "excluded binary extension"))
        if any(part in {"raw", "private", "__pycache__"} for part in relative.parts):
            issues.append((str(relative), "private, raw, or cache directory"))
        if path.suffix.lower() in TEXT_EXTENSIONS or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in LEAK_PATTERNS.items():
                if pattern.search(text):
                    issues.append((str(relative), label))

    print(f"Checked {len(files)} tracked files; required public sources present; {len(issues)} issues")
    for relative, reason in issues:
        print("WARN", relative, reason)
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
