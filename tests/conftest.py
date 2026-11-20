"""Gemeinsame Fixtures: die Mini-Reihe, an der jeder Schätzer von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture
def mini():
    """20 Beobachtungen, Schwelle 1: Überschreitungen an den Stellen 3, 4, 5, 12, 13, 19 (1-basiert), sonst 0. Von Hand: Abstände 1, 1, 7, 1, 6."""
    x = np.zeros(20)
    for pos in (3, 4, 5, 12, 13, 19):
        x[pos - 1] = 2.0
    return x, 1.0
