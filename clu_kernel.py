"""Der Gitterkern der Lindley-Kette (nur numpy): die Wahrheit, gegen die die Seite schätzt. Kein Schätzverfahren der Seite.

Zustand 0 (Atom) und Zellen ((j−1)h, jh] der Breite h, Vertreter die Zellmitte; Übergangswahrscheinlichkeiten aus der exakten Verteilungsfunktion von Z = S − A
(S ~ Exp(1), A ~ Exp(ρ)). Daraus:
* die stationäre Verteilung der Gitterkette (Rest oberhalb m·h auf den letzten Zustand gelegt),
* die exakte Verteilung des Tagesmaximums P(max_{i≤n} W_i ≤ x) = π Q^(n−1) 1 (Q: Kern auf den Zuständen bis x, stationär gestartet),
* der Extremalindex zur Schwelle u: θ(u) = (1 − λ_u)/F̄_h(u) mit dem Hauptwert λ_u des bei u abgetöteten Kerns (P(max W_1..W_n ≤ u) ≈ c λ_u^n).

Der Kern ist ein Gitter, kein exakter Wert im strengen Sinn: die Tests prüfen ihn gegen die geschlossene Randverteilung, gegen eine zweite Gitterweite und gegen die Simulation.
Der Grenzwert u → ∞ ist θ = (1 − ρ)² (Zerlegungsformel `theta_exact`; `tests/test_oracle_theta.py` rechnet sie in Brüchen nach)."""

import math

import numpy as np

import clu_queue as Q


# ------------------------------------------------------------------------------------------------------------------------------------------
# Geschlossene Formeln (Vehikel A)
# ------------------------------------------------------------------------------------------------------------------------------------------
def theta_exact(rho: float) -> float:
    """Extremalindex der M/M/1-Wartezeit im Grenzwert u → ∞: θ = (1 − ρ)² (eigene Herleitung, README Abschnitt Quellen)."""
    Q.check_rho(rho)
    return (1.0 - rho) ** 2


def mean_cluster_size(rho: float) -> float:
    """Mittlere Clustergröße im Grenzwert: 1/θ Kunden."""
    return 1.0 / theta_exact(rho)


def gumbel_day_cdf(rho: float, n: int, x, theta: float | None = None):
    """Gumbel-Näherung P(max ≤ x) ≈ exp(−n θ ρ e^{−γx}); θ = (1 − ρ)² (Standard) oder θ = 1 (die Annahme „unabhängig“)."""
    th = theta_exact(rho) if theta is None else theta
    x = np.asarray(x, dtype=float)
    return np.exp(-n * th * rho * np.exp(-(1.0 - rho) * x))


def gumbel_day_quantile(rho: float, n: int, q: float, theta: float | None = None) -> float:
    """Quantil der Gumbel-Näherung: x_q = (ln(n θ ρ) − ln(−ln q))/γ (kann negativ sein: die Näherung taugt dann nicht)."""
    th = theta_exact(rho) if theta is None else theta
    return (math.log(n * th * rho) - math.log(-math.log(q))) / (1.0 - rho)


def gumbel_shift(rho: float) -> float:
    """Grenzwert der Verschiebung durch die Annahme „unabhängig“: ln(1/θ)/γ Abfertigungsdauern."""
    return math.log(1.0 / theta_exact(rho)) / (1.0 - rho)


def clusters_among(k: int, theta: float) -> float:
    """Rechnerische Zahl unabhängiger Cluster unter k Überschreitungen: k·θ."""
    return k * theta


# ------------------------------------------------------------------------------------------------------------------------------------------
# Der Kern
# ------------------------------------------------------------------------------------------------------------------------------------------
def cdf_z(z, rho: float):
    """Verteilungsfunktion von Z = S − A: z ≥ 0: 1 − ρ/(1+ρ) e^{−z}; z < 0: e^{ρz}/(1+ρ)."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 - rho / (1.0 + rho) * np.exp(-np.maximum(z, 0.0)), np.exp(rho * np.minimum(z, 0.0)) / (1.0 + rho))


def kernel(rho: float, h: float, m: int) -> np.ndarray:
    """Übergangsmatrix T ((m+1)×(m+1)) auf den Zuständen 0..m; Zustand j ≥ 1 ist die Zelle ((j−1)h, jh]. Zeilensumme < 1: der Rest ist Masse über m·h."""
    Q.check_rho(rho)
    rep = np.concatenate([[0.0], (np.arange(1, m + 1) - 0.5) * h])
    edges = np.arange(0, m + 1) * h
    T = np.empty((m + 1, m + 1))
    T[:, 0] = cdf_z(-rep, rho)
    F = cdf_z(edges[None, :] - rep[:, None], rho)
    T[:, 1:] = F[:, 1:] - F[:, :-1]
    return T


def stationary(T: np.ndarray) -> np.ndarray:
    """Stationäre Verteilung der Gitterkette (Rest oberhalb auf den letzten Zustand gelegt); m muss weit über den betrachteten Niveaus liegen."""
    m1 = T.shape[0]
    Tl = T.copy()
    Tl[:, -1] += 1.0 - Tl.sum(axis=1)
    A = Tl.T - np.eye(m1)
    A[-1, :] = 1.0
    rhs = np.zeros(m1)
    rhs[-1] = 1.0
    return np.linalg.solve(A, rhs)


class Grid:
    """Kern, stationäre Verteilung und Überlebensfunktion auf einem Gitter (h, m) für eine Last."""

    def __init__(self, rho: float, h: float, umax: float):
        self.rho, self.h = Q.check_rho(rho), h
        self.m = int(round(umax / h))
        self.T = kernel(rho, h, self.m)
        self.pi = stationary(self.T)
        self.surv = np.concatenate([np.cumsum(self.pi[::-1])[::-1][1:], [0.0]])  # surv[j] = P(Zustand > j)

    def index(self, x: float) -> int:
        j = int(round(x / self.h))
        if j < 0 or j > self.m:
            raise ValueError("x liegt außerhalb des Gitters")
        return j

    def day_max_cdf(self, x: float, n: int) -> float:
        """P(max_{i≤n} W_i ≤ x), x auf eine Gitterkante gerundet, stationär gestartet."""
        j = self.index(x)
        v = self.pi[: j + 1].copy()
        Qm = self.T[: j + 1, : j + 1]
        for _ in range(n - 1):
            v = v @ Qm
        return float(v.sum())

    def day_max_cdf_at(self, x: float, ns) -> list[float]:
        """Wie `day_max_cdf` für mehrere Tageslängen in einem Durchgang (aufsteigend sortiert)."""
        j = self.index(x)
        v = self.pi[: j + 1].copy()
        Qm = self.T[: j + 1, : j + 1]
        out, step = [], 1
        for n in ns:
            while step < n:
                v = v @ Qm
                step += 1
            out.append(float(v.sum()))
        return out

    def principal(self, j: int, tol: float = 1e-15, max_iter: int = 200000) -> float:
        """Hauptwert des bei j·h abgetöteten Kerns (Potenzverfahren; die Matrix ist nichtnegativ, Perron-Frobenius)."""
        Qm = self.T[: j + 1, : j + 1]
        v = self.pi[: j + 1].copy()
        v /= v.sum()
        lam_old = 0.0
        lam = 0.0
        for _ in range(max_iter):
            w = v @ Qm
            lam = float(w.sum())
            v = w / lam
            if abs(lam - lam_old) < tol:
                break
            lam_old = lam
        return lam

    def theta_at(self, u: float) -> float:
        """θ(u) = (1 − λ_u)/F̄_h(u) an der Gitterkante nahe u."""
        j = self.index(u)
        return (1.0 - self.principal(j)) / float(self.surv[j])


def grid_for(rho: float, h: float, f_bar_min: float = 1e-4, margin: float = 60.0) -> Grid:
    """Gitter, das bis über die Schwelle zu F̄ = f_bar_min reicht (plus Rand `margin`: bei 6 verfälscht der letzte Zustand θ(u) bei F̄ = 10⁻³ um 32 % (Last 0,8) bis 99 % (Last 0,9), bei 60 ist θ(u) auf drei Stellen stabil, gemessen bei Last 0,5 bis 0,9)."""
    return Grid(rho, h, Q.mm1_threshold(rho, f_bar_min) + margin)


def theta_of_threshold(rho: float, f_bar: float, h: float = 0.2) -> float:
    """θ(u) zur Schwelle u mit P(W > u) = F̄ aus dem Kern (Gitter h). Rechenzeit: Sekundenbruchteile bis eine Sekunde bei Last 0,9 und F̄ = 10⁻⁴."""
    g = grid_for(rho, h, f_bar)
    return g.theta_at(Q.mm1_threshold(rho, f_bar))


def day_max_quantile(xs, cdf, q: float) -> float:
    """Quantil aus einer Verteilungsfunktion auf Gitterpunkten xs (aufsteigend, cdf nichtfallend) durch lineare Interpolation; nan außerhalb des Bereichs."""
    xs = np.asarray(xs, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    if not (cdf[0] <= q <= cdf[-1]):
        return math.nan
    return float(np.interp(q, cdf, xs))
