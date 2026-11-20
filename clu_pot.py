"""Schwellenweg mit Abhängigkeit (nur numpy und Standardbibliothek): GPD-Anpassung, Pegel und das Delta-Intervall mit θ-Korrektur.

Konvention (einmal festgelegt): GPD(ξ, σ) mit Überlebensfunktion (1 + ξ y/σ)^(−1/ξ) für ξ ≠ 0 und exp(−y/σ) für ξ = 0; ξ > 0 schwerer Rand, ξ = 0 Gumbel-Typ, ξ < 0 endlicher Endpunkt.
`scipy.stats.genpareto` hat c = ξ, `scipy.stats.genextreme` dagegen c = −ξ; beides prüfen nur die Tests, die App ruft scipy nie auf.

Der Pegel x_p gehört zur Überschreitungswahrscheinlichkeit p JE BEOBACHTUNG (hier je Kunde): x_p = u + (σ/ξ)((ζ/p)^ξ − 1), ζ = k/n; für ξ = 0 u + σ ln(ζ/p).
Das Delta-Intervall gilt für ln x_p (erwartete Information der GPD nach Smith 1987, Gestalt aus dem Gedächtnis, Stufe G, das Vorzeichen des Nebenglieds an einer Simulation geprüft; binomiale Varianz für ζ). Bei serieller Abhängigkeit wird die
gesamte Varianz durch θ geteilt (k Überschreitungen zählen wie k·θ unabhängige Cluster); die Wahl der Skala (ln) und des Anteils, der durch θ geteilt wird, ist eine Konstruktionswahl, keine Eigenschaft der Methode.
Gemessen: bei M/M/1-Abhängigkeit streut der Pegel stärker, als dieses Teilen annimmt (siehe README, Experiment 3)."""

import math

import numpy as np

XI_MIN = -1.0       # Nebenbedingung der ML: für ξ < −1 ist die Likelihood unbeschränkt
Z95 = 1.959964      # 97,5-%-Punkt der Standardnormalverteilung


def pot_level(xi: float, sigma: float, u: float, zeta: float, p: float) -> float:
    """Pegel mit Überschreitungswahrscheinlichkeit p je Beobachtung; zeta = Anteil der Beobachtungen über u."""
    ratio = zeta / p
    if abs(xi) < 1e-8:
        return u + sigma * math.log(ratio)
    return u + sigma / xi * (ratio ** xi - 1.0)


def _golden_max(f, a: float, b: float, iters: int = 80) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    best_x, best_f = (c, fc) if fc >= fd else (d, fd)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
            if fc > best_f:
                best_x, best_f = c, fc
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
            if fd > best_f:
                best_x, best_f = d, fd
        if abs(b - a) < 1e-12 * (abs(c) + abs(d) + 1e-12):
            break
    return best_x


def gpd_ml(y, xi_min: float = XI_MIN):
    """Maximum Likelihood der GPD (ξ, σ) an Überschreitungen y > 0 über das Profil in t = ξ/σ · mean(y); gibt (ξ̂, σ̂, Randtreffer).

    Randtreffer: die Lösung liegt an der Wand ξ = ξ_min oder am oberen Ende des Suchgitters. Ohne die Nebenbedingung wäre die Likelihood für ξ < −1 unbeschränkt."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        raise ValueError("Die Anpassung braucht mindestens zwei Überschreitungen")
    yb = float(y.mean())
    ymax = float(y.max())
    tlo = -yb / ymax * (1.0 - 1e-9)
    r = y / yb

    def xi_of(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.mean(np.log1p(t[:, None] * r[None, :]), axis=1)

    def ll(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        xi = xi_of(t)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = n * np.log(t / xi) - n * math.log(yb) - n * (xi + 1.0)
        bad = ~np.isfinite(val) | (xi < xi_min) | (xi == 0.0) | (t / np.where(xi == 0.0, 1.0, xi) <= 0)
        val = np.where(bad, -np.inf, val)
        val = np.where(np.abs(t) < 1e-9, -n * math.log(yb) - n, val)
        return val

    grid = np.concatenate([np.linspace(tlo, -1e-3, 70), [0.0], np.geomspace(1e-3, 40.0, 90)])
    vals = ll(grid)
    i = int(np.argmax(vals))
    if not np.isfinite(vals[i]):
        return math.nan, math.nan, True
    a, b = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, len(grid) - 1)])
    t, best = float(grid[i]), float(vals[i])
    if a < b:
        fine = np.linspace(a, b, 41)
        fv = ll(fine)
        j = int(np.argmax(fv))
        if fv[j] > best:
            t, best = float(fine[j]), float(fv[j])
        a2, b2 = float(fine[max(j - 1, 0)]), float(fine[min(j + 1, len(fine) - 1)])
        t_opt = _golden_max(lambda s: float(ll(s)[0]), a2, b2)
        if float(ll(t_opt)[0]) >= best - 1e-12:
            t = t_opt
    if abs(t) < 1e-9:
        return 0.0, yb, False
    xi = float(xi_of(t)[0])
    sigma = xi * yb / t
    lower_edge = xi <= xi_min + 1e-3
    if xi_min == -1.0 and (lower_edge or -n * math.log(ymax) > float(ll(t)[0]) + 1e-12):
        # Am Rand ξ = −1 ist die Dichte (1/σ) je Wert, solange σ ≥ größte Überschreitung; das beschränkte Maximum liegt dort bei σ = max(y).
        xi, sigma, lower_edge = -1.0, ymax, True
    return xi, sigma, bool(lower_edge or i >= len(grid) - 2)


def gpd_cov(xi: float, sigma: float, k: int):
    """Asymptotische Kovarianzmatrix von (σ̂, ξ̂) der GPD-Maximum-Likelihood bei k unabhängigen Überschreitungen: (1+ξ)/k · [[2σ², −σ], [−σ, 1+ξ]].

    Das Nebenglied ist NEGATIV (Smith 1987; Simulation in `tests/test_oracle_gpd.py`: Kovarianz bei ξ = 0, σ = 1, k = 600 etwa −0,0018 gegen −1/k = −0,0017). Eine frühere Fassung dieser Demo hatte +σ(1+ξ)/k."""
    return (1 + xi) / k * np.array([[2 * sigma * sigma, -sigma], [-sigma, 1 + xi]])


def delta_ci_logx(xi: float, sigma: float, u: float, k: int, nobs: int, p: float, theta: float = 1.0, z: float = Z95):
    """Delta-Intervall (lo, hi) für ln des Pegels zur Überschreitungswahrscheinlichkeit p je Beobachtung.

    Die gesamte Varianz wird durch `theta` geteilt: theta = 1 ist das Intervall für unabhängige Beobachtungen, theta < 1 zählt k Überschreitungen wie k·theta Cluster."""
    if theta <= 0:
        raise ValueError("θ muss positiv sein")
    zeta = k / nobs
    V = gpd_cov(xi, sigma, k)
    vz =zeta * (1 - zeta) / nobs

    def f(par):
        return math.log(pot_level(par[2], par[1], u, par[0], p))

    par0 = np.array([zeta, sigma, xi])
    g = np.zeros(3)
    for j, h in enumerate([zeta * 1e-4, sigma * 1e-5, 1e-6]):
        pp, pm = par0.copy(), par0.copy()
        pp[j] += h
        pm[j] -= h
        g[j] = (f(pp) - f(pm)) / (2 * h)
    var = (g[0] ** 2 * vz + g[1:] @ V @ g[1:]) / theta
    c = f(par0)
    se = math.sqrt(max(var, 0.0))
    return c - z * se, c + z * se


def fit_threshold(x, frac: float):
    """Schwelle u = empirisches (1 − frac)-Quantil, k = Zahl der Werte über u; gibt (u, k, y) mit den Überschreitungen y = x − u."""
    x = np.asarray(x, dtype=float)
    u = float(np.quantile(x, 1.0 - frac))
    over = x[x > u]
    return u, int(over.size), over - u


def fit_topk(x, k: int):
    """Schwelle u = (k+1)-größter Wert und die k Überschreitungen y = x − u."""
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    if not (1 <= k < n):
        raise ValueError("k muss zwischen 1 und n − 1 liegen")
    u = float(xs[n - k - 1])
    return u, k, xs[n - k:] - u
