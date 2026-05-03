#!/usr/bin/env python3
"""Backward-compatible entrypoint for preprocessing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from projet_dl.cli.preprocess import main


if __name__ == "__main__":
    main()
