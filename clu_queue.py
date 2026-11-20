"""Die Reihen der Demo: M/M/1-Wartezeit (Lindley-Rekursion) und max-AR(1). Nur numpy und Standardbibliothek.

Zufall: der Zufallsgenerator der Vorlage (`clu_rng.SplitMix64`, wörtliche Kopie). Weil die Reihen lang sind (bis 20.000 Kunden je Lauf, bis 12.000 in der Abdeckungsstudie), wird der Strom
in Zähler-Form erzeugt: der k-te Ausgang von `SplitMix64(seed)` ist mix(seed + k·γ), als numpy-Vektor also ohne Schleife zu haben. Die Gleichverteilten sind bitgleich zu
`SplitMix64.random()` (Test `tests/test_rng_stream.py`); die Exponentialzahlen sind -mean·ln(1-u) mit `numpy.log` statt der Dezimal-Rechnung der Vorlage und weichen von
`SplitMix64.expovariate` höchstens in der letzten Binärstelle ab (Test: relativ 1e-12). Zufallsanteile der README sind darum Bänder, nie ein Test auf das letzte Bit.

Modell Vehikel A: M/M/1, Dienstrate 1 (die Abfertigungsdauer ist die Zeiteinheit), Ankunftsrate ρ (Last), Wartezeit W_{n+1} = max(0, W_n + S_n − A_{n+1}),
S ~ Exp(1), A ~ Exp(ρ); γ = 1 − ρ. Stationär gestartet: W_1 = 0 mit Wahrscheinlichkeit 1 − ρ, sonst Exp(γ) (exakt die stationäre Verteilung).
Modell max-AR(1): X_t = max(φ X_{t−1}, (1−φ) Z_t), Z ~ Fréchet(1), X_1 = Z_1; Rand Fréchet(1), θ = 1 − φ."""

import math

import numpy as np

import clu_rng

_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
MASK64 = (1 << 64) - 1


# ------------------------------------------------------------------------------------------------------------------------------------------
# Strom in Zähler-Form
# ------------------------------------------------------------------------------------------------------------------------------------------
def sm64_u64(seed: int, n: int, start: int = 0) -> np.ndarray:
    """n Ausgänge ab dem (start + 1)-ten, wie `SplitMix64(seed)` nach `start` Aufrufen von `next_u64`."""
    with np.errstate(over="ignore"):
        k = np.arange(start + 1, start + n + 1, dtype=np.uint64)
        z = np.uint64(seed & MASK64) + k * _G
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def sm64_unif(seed: int, n: int, start: int = 0) -> np.ndarray:
    """Gleichverteilt in [0, 1) mit 53 Bit, wie `SplitMix64.random()`."""
    return (sm64_u64(seed, n, start) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def exp_from_u(u, mean: float = 1.0):
    """Exponentialverteilt mit dem Mittel `mean` aus Gleichverteilten u in [0, 1): −mean · ln(1 − u)."""
    return -mean * np.log(1.0 - np.asarray(u, dtype=float))


def stream_seeds(seed: int) -> tuple[int, int, int]:
    """Drei getrennte Ströme je Zufalls-Seed: Start (stationäre Verteilung), Bedienzeiten S, Zwischenankunftszeiten A."""
    return (seed * 3) & MASK64, (seed * 3 + 1) & MASK64, (seed * 3 + 2) & MASK64


# ------------------------------------------------------------------------------------------------------------------------------------------
# M/M/1-Wartezeit
# ------------------------------------------------------------------------------------------------------------------------------------------
def check_rho(rho: float) -> float:
    if not (0.0 < rho < 1.0):
        raise ValueError("Die Last ρ muss zwischen 0 und 1 liegen (ρ ≥ 1: die Schlange ist nicht stationär)")
    return float(rho)


def mm1_wait(rho: float, n: int, seed: int, run: int = 0) -> np.ndarray:
    """Wartezeiten von n aufeinanderfolgenden Kunden einer stationär gestarteten M/M/1-Schlange (Dienstrate 1, Ankunftsrate ρ).

    Lauf Nummer `run` desselben Zufalls-Seeds nimmt ein zusammenhängendes Stück jedes der drei Ströme (zwei Gleichverteilte für den Start, n − 1 je Bedienzeit und Zwischenankunft).
    Die Rekursion W_{i+1} = max(0, W_i + S_i − A_{i+1}) wird über C_i = W_1 + Σ_{j<i} (S_j − A_{j+1}) und W_i = C_i − min(0, min_{j≤i} C_j) ohne Schleife ausgewertet
    (Skorokhod-Reflexion; der Test vergleicht mit der Schleife der Definition)."""
    rho = check_rho(rho)
    if n < 1:
        raise ValueError("n muss mindestens 1 sein")
    gamma = 1.0 - rho
    s0, s1, s2 = stream_seeds(seed)
    u0 = sm64_unif(s0, 2, 2 * run)
    w1 = float(exp_from_u(u0[1], 1.0 / gamma)) if u0[0] < rho else 0.0
    m = n - 1
    s = exp_from_u(sm64_unif(s1, m, run * n), 1.0)
    a = exp_from_u(sm64_unif(s2, m, run * n), 1.0 / rho)
    c = np.empty(n)
    c[0] = w1
    c[1:] = w1 + np.cumsum(s - a)
    return c - np.minimum(0.0, np.minimum.accumulate(c))


def mm1_wait_loop(rho: float, n: int, seed: int, run: int = 0) -> list[float]:
    """Dieselbe Reihe nach der Definition mit einer Schleife und den Strömen der Vorlage (`clu_rng.SplitMix64`, `expovariate` über die Dezimal-Rechnung): zweiter Rechenweg für den Test."""
    rho = check_rho(rho)
    gamma = 1.0 - rho
    s0, s1, s2 = stream_seeds(seed)
    r0, rs, ra = clu_rng.SplitMix64(s0), clu_rng.SplitMix64(s1), clu_rng.SplitMix64(s2)
    for _ in range(2 * run):
        r0.next_u64()
    for _ in range(run * n):
        rs.next_u64()
        ra.next_u64()
    u_start = r0.random()
    w = r0.expovariate(1.0 / gamma) if u_start < rho else 0.0
    out = [w]
    for _ in range(n - 1):
        w = max(0.0, w + rs.expovariate(1.0) - ra.expovariate(1.0 / rho))
        out.append(w)
    return out


def mm1_level(rho: float, p: float) -> float:
    """Pegel x_p mit P(W > x_p) = p je Kunde: x_p = ln(ρ/p)/γ (exakt, ρ > p)."""
    rho = check_rho(rho)
    if not (0.0 < p < rho):
        raise ValueError("p muss zwischen 0 und ρ liegen (P(W > 0) = ρ)")
    return math.log(rho / p) / (1.0 - rho)


def mm1_threshold(rho: float, f_bar: float) -> float:
    """Schwelle u mit P(W > u) = F̄ (Randanteil über der Schwelle): u = ln(ρ/F̄)/γ."""
    return mm1_level(rho, f_bar)


def mm1_sf(rho: float, x):
    """P(W > x) = ρ e^{−γx} für x ≥ 0 (stationär)."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, rho * np.exp(-(1.0 - rho) * np.maximum(x, 0.0)), 1.0)


# ------------------------------------------------------------------------------------------------------------------------------------------
# max-AR(1) mit Fréchet-Rand
# ------------------------------------------------------------------------------------------------------------------------------------------
def frechet_from_u(u):
    """Fréchet(1) aus Gleichverteilten: −1/ln(u), u unten auf 1e-300 begrenzt (u = 0 hat die Wahrscheinlichkeit 2^−53)."""
    return -1.0 / np.log(np.maximum(np.asarray(u, dtype=float), 1e-300))


def maxar1(phi: float, n: int, seed: int, run: int = 0) -> np.ndarray:
    """n aufeinanderfolgende Werte des max-AR(1)-Prozesses X_t = max(φ X_{t−1}, (1−φ) Z_t), X_1 = Z_1, Z ~ Fréchet(1) (stationär: Rand Fréchet(1))."""
    if not (0.0 <= phi < 1.0):
        raise ValueError("φ muss in [0, 1) liegen")
    z = frechet_from_u(sm64_unif(seed & MASK64, n, run * n))
    x = np.empty(n)
    x[0] = z[0]
    zs = (1.0 - phi) * z
    prev = z[0]
    for t in range(1, n):
        prev = max(phi * prev, zs[t])
        x[t] = prev
    return x


def maxar1_block_scale(phi: float, b: int) -> float:
    """Das Maximum über b aufeinanderfolgende Werte ist Fréchet(1) mit der Skala s_b = 1 + (b − 1)(1 − φ): P(M_b ≤ x) = exp(−s_b/x)."""
    return 1.0 + (b - 1) * (1.0 - phi)


def maxar1_block_cdf(phi: float, b: int, x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(x > 0, np.exp(-maxar1_block_scale(phi, b) / np.where(x > 0, x, 1.0)), 0.0)


def frechet_level(p: float) -> float:
    """Fréchet(1)-Quantil zur Überschreitungswahrscheinlichkeit p je Beobachtung: x = −1/ln(1 − p)."""
    return -1.0 / math.log1p(-p)
