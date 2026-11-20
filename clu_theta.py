"""Schätzer des Extremalindex θ und die Zerlegung in Cluster (nur numpy). θ ist der Extremalindex, nicht der Extremwertindex ξ.

Alle Schätzer arbeiten an den Stellen der Überschreitungen einer festen Schwelle u:
* Runs-Schätzer: zwei Überschreitungen gehören zum selben Cluster, wenn ihr Abstand höchstens r Kunden beträgt; θ̂ = Zahl der Cluster / Zahl der Überschreitungen.
* Blöcke-Schätzer: θ̂ = ln(1 − K/B) / (b ln(1 − N/(B b))) mit B Blöcken der Länge b, K Blöcken mit Überschreitung und N Überschreitungen insgesamt.
* Intervall-Schätzer (Ferro/Segers 2003): aus den Abständen T_i der Überschreitungen; Formel aus dem Gedächtnis (Stufe G), hier an exakten Werten geprüft (Tests und README).
Zähleinheit der Rechnung: Überschreitungen (N) und Cluster; „unabhängige Cluster“ heißt k·θ."""

import math

import numpy as np

R_DEFAULT = 10
B_DEFAULT = 50


def exceedance_positions(x, u: float) -> np.ndarray:
    """Stellen (0-basiert) der Beobachtungen über der Schwelle u (streng größer)."""
    return np.flatnonzero(np.asarray(x) > u)


def clusters(x, u: float, r: int) -> list[np.ndarray]:
    """Cluster der Überschreitungen nach der Runs-Regel: ein neuer Cluster beginnt, wenn der Abstand zur vorigen Überschreitung größer als r ist."""
    if r < 1:
        raise ValueError("r muss mindestens 1 sein")
    idx = exceedance_positions(x, u)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > r) + 1
    return [part for part in np.split(idx, cuts)]


def theta_runs(x, u: float, r: int = R_DEFAULT) -> float:
    """Runs-Schätzer; nan ohne Überschreitung."""
    idx = exceedance_positions(x, u)
    if idx.size == 0:
        return math.nan
    return (1 + int(np.sum(np.diff(idx) > r))) / idx.size


def theta_blocks(x, u: float, b: int = B_DEFAULT) -> float:
    """Blöcke-Schätzer; nan, wenn kein Block oder jeder Block eine Überschreitung hat (der Logarithmus wäre nicht definiert) oder weniger als ein Block da ist."""
    x = np.asarray(x)
    B = len(x) // b
    if B < 1:
        return math.nan
    xx = x[: B * b]
    N = int(np.sum(xx > u))
    K = int(np.sum(xx.reshape(B, b).max(axis=1) > u))
    if K == 0 or K == B or N == 0 or N == B * b:
        return math.nan
    return math.log(1 - K / B) / (b * math.log(1 - N / (B * b)))


def theta_intervals(x, u: float) -> float:
    """Intervall-Schätzer nach Ferro und Segers (2003), auf höchstens 1 begrenzt; nan bei weniger als 3 Überschreitungen."""
    idx = exceedance_positions(x, u)
    N = idx.size
    if N < 3:
        return math.nan
    T = np.diff(idx).astype(float)
    if T.max() <= 2:
        th = 2 * T.sum() ** 2 / ((N - 1) * np.sum(T * T))
    else:
        th = 2 * np.sum(T - 1) ** 2 / ((N - 1) * np.sum((T - 1) * (T - 2)))
    return min(1.0, th)


def mean_cluster_size_hat(theta_hat: float) -> float:
    """Kehrwert des Schätzers: mittlere Clustergröße in Überschreitungen je Cluster."""
    return math.nan if not (theta_hat > 0) else 1.0 / theta_hat
