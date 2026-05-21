#!/usr/bin/env python3
"""Rewrite the PyPI distribution name in pyproject.toml.

Used by .github/workflows/wheels.yml so that a fork can publish under a
different project name (e.g. ``easypytorch3d`` or ``easypytorch3d-cu130``)
while keeping the importable package name ``pytorch3d`` untouched.

Usage:
    python scripts/rename_distribution.py easypytorch3d-cu130
"""
from __future__ import annotations

import pathlib
import re
import sys


def rewrite(new_name: str) -> None:
    root = pathlib.Path(__file__).resolve().parents[1]

    pyproject = root / "pyproject.toml"
    text = pyproject.read_text()
    new_text, count = re.subn(
        r'^(name\s*=\s*)"pytorch3d"',
        rf'\1"{new_name}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(
            f'Could not find `name = "pytorch3d"` in {pyproject}'
        )
    pyproject.write_text(new_text)
    print(f"Renamed distribution to: {new_name}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    rewrite(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
