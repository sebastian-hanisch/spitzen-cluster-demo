"""Lädt die eingefrorenen Messungen (Quelle und Stand stehen in jeder Datei; Erzeuger und Prüfung: `tools/clu_make_*.py`, je mit `--check`).

* `clu_exact.json`: exakte Verteilung des Tagesmaximums (Last 0,5 / 0,8 / 0,9, n = 16 bis 1.024 Kunden) und θ(u) aus dem Kern. Kein Zufall.
* `clu_blocks.json`: Populations-ξ der Blockmaxima (beste GEV nach Kolmogorov-Abstand und nach Likelihood), Last 0,5 und 0,8, Blocklänge 64 bis 4.096 Kunden. Kein Zufall.
* `clu_coverage.json`: Abdeckung der 95-%-Intervalle (Experiment 3), Schätzer von θ (Experiment 1) und Block gegen Schwelle am max-AR(1) (Experiment 5), Zufall aus SplitMix64."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def exact() -> dict:
    return _load("clu_exact.json")


def blocks() -> dict:
    return _load("clu_blocks.json")


def coverage() -> dict:
    return _load("clu_coverage.json")


# ------------------------------------------------------------------------------------------------------------------------------------------
def day_cdf(rho: float, n: int):
    """Exakte Verteilungsfunktion des Tagesmaximums auf dem Raster der Breite 0,5: (x, F(x)); nur der Bereich 1e-6 ≤ F ≤ 1 − 1e-6."""
    d = exact()["tag"][str(rho)]["n"][str(n)]
    xs = d["x0"] + exact()["raster_x"] * np.arange(len(d["cdf"]))
    return xs, np.array(d["cdf"])


def day_quantile(rho: float, n: int, q: float) -> float:
    """Exaktes Quantil des Tagesmaximums (Feinraster 0,1 um die grobe Schätzung, linear interpoliert; Gitterfehler h = 0,1)."""
    return exact()["tag"][str(rho)]["quantile"][str(n)][str(q)]


def theta_u(rho: float, f_bar: float, h: float = 0.1) -> dict:
    """θ(u) aus dem Kern zur Schwelle mit Randanteil F̄: {'u', 'theta', 'f_bar_gitter'}."""
    return exact()["theta_u"][str(rho)][str(f_bar)][str(h)]


def block_row(rho: float, n: int) -> dict:
    return blocks()["daten"][str(rho)][str(n)]


def coverage_cell(rho: float, n: int, frac: float, p: float) -> dict:
    for c in coverage()["abdeckung"]:
        if c["rho"] == rho and c["n"] == n and c["anteil"] == frac and c["p"] == p:
            return c
    raise KeyError((rho, n, frac, p))


def estimator_rows() -> list:
    return coverage()["schaetzer"]


def block_schwelle_row(phi: float) -> dict:
    for r in coverage()["block_schwelle"]:
        if r["phi"] == phi:
            return r
    raise KeyError(phi)
