#!/usr/bin/env python3
"""Collect Isaac Lab height-scan alignment samples for gx-real.

Run this through the Isaac Lab Python launcher when collecting real simulation
samples. It delegates contract writing and validation to
``export_height_scan_contract.py`` with sample collection enabled.
"""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_height_scan_contract import main  # noqa: E402


if __name__ == "__main__":
    main(default_collect_samples=True)
