"""Orakel für den Schwellenweg: GPD-Anpassung, Pegel und Delta-Intervall gegen scipy und gegen eine zweite Ableitung (scipy nur hier, nie in der App).

Konvention: `scipy.stats.genpareto` hat c = ξ, `scipy.stats.genextreme` dagegen c = −ξ. (a) Die Likelihood der Demo-Lösung ist nie schlechter als die von scipy und die Lösung ein Stationärpunkt;
(b) der Pegel gegen `genpareto.isf`; (c) das Delta-Intervall gegen eine unabhängige Fassung mit Complex-Step-Ableitung (teilt die Matrix mit dem Code; das Vorzeichen des Nebenglieds und die Abdeckung prüfen Simulationen am Ende der Datei); (d) die eingefrorene Populationsanpassung gegen `genextreme` mit c = −ξ."""

import math

import numpy as np
import pytest

import clu_pot as P
import clu_queue as Q

sp = pytest.importorskip("scipy.stats")


def _sample(xi: float, k: int, seed: int, sigma: float = 1.0) -> np.ndarray:
    u = Q.sm64_unif(seed, k)
    v = 1.0 - u
    return -sigma * np.log(v) if xi == 0 else sigma * (v ** (-xi) - 1.0) / xi


def _nll(y, xi, sg):
    if sg <= 0 or xi < -1 - 1e-12:
        return math.inf
    if xi <= -1 + 1e-12:                      # Wand ξ = −1: Dichte 1/σ je Wert, solange σ ≥ größte Überschreitung
        return float(len(y) * math.log(sg)) if sg >= float(y.max()) else math.inf
    t = 1 + xi * y / sg
    if np.any(t <= 0):
        return math.inf
    if abs(xi) < 1e-10:
        return float(len(y) * math.log(sg) + y.sum() / sg)
    return float(len(y) * math.log(sg) + (1 + 1 / xi) * np.sum(np.log(t)))


CASES = [(xi, k, seed) for xi in (-0.2, 0.0, 0.3) for k in (30, 100, 400) for seed in (1, 2, 3, 4)]


@pytest.mark.parametrize("xi,k,seed", CASES)
def test_gpd_ml_ist_mindestens_so_gut_wie_scipy_und_ein_stationaerpunkt(xi, k, seed):
    y = _sample(xi, k, 1000 * seed + k)
    xh, sh, edge = P.gpd_ml(y)
    c, loc, sc = sp.genpareto.fit(y, floc=0)                                        # c = ξ
    assert _nll(y, xh, sh) <= _nll(y, c, sc) + 1e-7 * k
    if not edge:
        for dx, ds in ((1e-4, 0), (-1e-4, 0), (0, 1e-4), (0, -1e-4)):
            assert _nll(y, xh + dx, sh + ds) >= _nll(y, xh, sh) - 1e-9                # lokales Minimum der negativen Log-Likelihood
        assert xh == pytest.approx(c, abs=2e-3) and sh == pytest.approx(sc, rel=2e-3)


def test_gpd_ml_exakt_exponentielle_ueberschreitungen_liefern_xi_nahe_null():
    xs = [P.gpd_ml(_sample(0.0, 150, s))[0] for s in range(40)]
    assert abs(np.mean(xs)) < 0.06 and np.std(xs, ddof=1) == pytest.approx(1.0 / math.sqrt(150), rel=0.35)


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.4])
def test_pegel_gleich_genpareto_isf(xi):
    u, zeta, sg, p = 7.5, 0.05, 1.7, 1e-3
    want = u + float(sp.genpareto.isf(p / zeta, c=xi, scale=sg))
    assert P.pot_level(xi, sg, u, zeta, p) == pytest.approx(want, rel=1e-12)


def _delta_complex_step(xi, sg, u, k, n, p, theta):
    zeta = k / n
    h = 1e-30

    def f(zt, s, x):
        r = zt / p
        return np.log(u + s / x * (r ** x - 1.0))

    g = np.array([f(zeta + 1j * h, sg, xi).imag / h, f(zeta, sg + 1j * h, xi).imag / h, f(zeta, sg, xi + 1j * h).imag / h])
    V = (1 + xi) / k * np.array([[2 * sg * sg, -sg], [-sg, 1 + xi]])
    var = (g[0] ** 2 * zeta * (1 - zeta) / n + g[1:] @ V @ g[1:]) / theta
    c = float(f(zeta, sg, xi).real)
    return c - 1.959964 * math.sqrt(var), c + 1.959964 * math.sqrt(var)


@pytest.mark.parametrize("xi,sg,k,n,theta", [(0.1, 1.0, 150, 3000, 1.0), (0.2, 2.0, 60, 3000, 0.25), (-0.1, 0.5, 300, 12000, 0.04), (0.3, 1.2, 50, 1000, 0.5)])
def test_delta_intervall_gleich_der_zweiten_ableitung_mit_complex_step(xi, sg, k, n, theta):
    u, p = 10.0, 1e-3
    lo, hi = P.delta_ci_logx(xi, sg, u, k, n, p, theta)
    lo2, hi2 = _delta_complex_step(xi, sg, u, k, n, p, theta)
    assert lo == pytest.approx(lo2, abs=1e-6) and hi == pytest.approx(hi2, abs=1e-6)


def test_delta_intervall_die_breite_waechst_mit_eins_durch_wurzel_theta():
    base = P.delta_ci_logx(0.1, 1.0, 10.0, 150, 3000, 1e-3, 1.0)
    for th in (0.5, 0.25, 0.04):
        lo, hi = P.delta_ci_logx(0.1, 1.0, 10.0, 150, 3000, 1e-3, th)
        assert (hi - lo) == pytest.approx((base[1] - base[0]) / math.sqrt(th), rel=1e-9)


def test_populationsanpassung_eingefroren_gleich_genextreme_mit_c_minus_xi():
    import clu_reference as REF
    for rho, n in ((0.8, 256), (0.8, 4096), (0.5, 64)):
        d = REF.block_row(rho, n)
        g = d["ks_gev"]
        x = np.array(d["x"])
        cdf = sp.genextreme.cdf(x, c=-g["xi"], loc=g["mu"], scale=g["sigma"])       # scipy: c = −ξ
        assert float(np.max(np.abs(cdf - np.array(d["cdf"])))) == pytest.approx(g["ks"], abs=2e-6)
        gg = d["ks_gumbel"]
        cdf0 = sp.gumbel_r.cdf(x, loc=gg["mu"], scale=gg["sigma"])
        assert float(np.max(np.abs(cdf0 - np.array(d["cdf"])))) == pytest.approx(gg["ks"], abs=2e-6)
        assert g["ks"] < gg["ks"]


# --- Das Vorzeichen des Nebenglieds und die Abdeckung des Delta-Intervalls: unabhängige Rechenwege (Simulation) ------------------------------------
# Die Complex-Step-Fassung oben teilt die Matrix mit dem Code und fängt ein falsches Vorzeichen nicht. Diese Tests rechnen gegen Simulation:
# (1) Kovarianz von (σ̂, ξ̂) an Exp(1)-Überschreitungen gegen −σ(1+ξ)/k; (2) Streuung und Abdeckung des Pegels an unabhängigen Beobachtungen aus der Randverteilung der M/M/1-Wartezeit.
def _iid_wait(rho: float, n: int, seed: int, run: int) -> np.ndarray:
    u = Q.sm64_unif(seed, 2 * n, 2 * n * run)
    return np.where(u[:n] < rho, -np.log(1.0 - u[n:]) / (1.0 - rho), 0.0)


def test_gpd_cov_hat_das_negative_nebenglied():
    V = P.gpd_cov(0.0, 1.0, 600)
    assert V[0, 1] == V[1, 0] == pytest.approx(-1 / 600) and V[0, 0] == pytest.approx(2 / 600) and V[1, 1] == pytest.approx(1 / 600)


def test_kovarianz_von_sigma_und_xi_gegen_simulation_k_600():
    k, R = 600, 1200
    fits = np.array([P.gpd_ml(_sample(0.0, k, 5000 + r))[:2] for r in range(R)])           # (ξ̂, σ̂)
    xs, ss = fits[:, 0], fits[:, 1]
    emp = np.cov(ss, xs)
    V = P.gpd_cov(0.0, 1.0, k)
    se = math.sqrt((emp[0, 0] * emp[1, 1] + emp[0, 1] ** 2) / (R - 1))
    assert emp[0, 1] < 0 and abs(emp[0, 1] - V[0, 1]) < max(4 * se, 0.25 * abs(V[0, 1]))      # endliches k: etwa 8 % größer als die asymptotische Matrix
    assert abs(emp[0, 1] - (-V[0, 1])) > 15 * se                                                # das frühere Vorzeichen (+) liegt um mehr als das Zehnfache des Standardfehlers daneben
    assert emp[0, 0] / V[0, 0] == pytest.approx(1.0, abs=0.2) and emp[1, 1] / V[1, 1] == pytest.approx(1.0, abs=0.2)


def _pegel_lauf(rho, n, frac, p, R, seed, wait):
    xt = math.log(Q.mm1_level(rho, p))
    err, se, hit = [], [], []
    for r in range(R):
        x = wait(rho, n, seed, r)
        u, k, y = P.fit_threshold(x, frac)
        xi, sg, _e = P.gpd_ml(y)
        lo, hi = P.delta_ci_logx(xi, sg, u, k, n, p, 1.0)
        c = math.log(P.pot_level(xi, sg, u, k / n, p))
        err.append(c - xt)
        se.append((hi - lo) / (2 * 1.959964))
        hit.append(lo <= xt <= hi)
    return np.array(err), np.array(se), np.array(hit)


def test_delta_intervall_bei_unabhaengigen_daten_streuung_und_abdeckung():
    """Unabhängige Beobachtungen (θ = 1), Last 50 %, n = 12.000, 5 % über der Schwelle (k = 600), p = 10⁻³: das Intervall hält etwa sein Versprechen.
    Mit dem früheren Vorzeichen waren die Intervalle mehr als doppelt so breit, wie die Streuung des Pegels verlangt (Verhältnis 0,46), und die Abdeckung lag bei 100 %."""
    err, se, hit = _pegel_lauf(0.5, 12000, 0.05, 1e-3, 300, 9000, _iid_wait)
    assert err.std(ddof=1) / se.mean() == pytest.approx(1.0, abs=0.15)
    assert abs(hit.mean() - 0.93) <= 0.035                                                    # README: rund 93 %


def test_delta_intervall_bei_unabhaengigen_daten_und_150_ueberschreitungen_deckt_etwas_zu_selten():
    err, se, hit = _pegel_lauf(0.5, 3000, 0.05, 1e-3, 500, 9100, _iid_wait)
    assert abs(hit.mean() - 0.89) <= 0.035 and err.std(ddof=1) / se.mean() == pytest.approx(1.1, abs=0.2)           # README: rund 89 %


def test_bei_abhaengigkeit_streut_die_gpd_anpassung_staerker_als_das_teilen_durch_theta_annimmt():
    """Last 50 % (θ = 1/4), n = 12.000, 5 % über der Schwelle: ln des Pegels streut etwa dreimal so stark wie bei unabhängigen Daten, nicht doppelt (1/√θ = 2)."""
    dep, _s, _h = _pegel_lauf(0.5, 12000, 0.05, 1e-3, 150, 9200, Q.mm1_wait)
    ind, _s2, _h2 = _pegel_lauf(0.5, 12000, 0.05, 1e-3, 150, 9300, _iid_wait)
    ratio = dep.std(ddof=1) / ind.std(ddof=1)
    assert 2.3 <= ratio <= 4.2


# --- Die erste Fassung (Nebenglied +) ist reproduzierbar: die Zahlen vor der Korrektur stehen in der README nur als Vergleich ----------------------
def _alte_matrix(xi, sigma, k):
    return (1 + xi) / k * np.array([[2 * sigma * sigma, sigma], [sigma, 1 + xi]])


def test_erste_fassung_mit_falschem_vorzeichen_reproduziert_die_alten_zahlen(monkeypatch):
    import clu_evaluation as E
    import clu_reference as REF
    monkeypatch.setattr(P, "gpd_cov", _alte_matrix)
    for rho, old in ((0.5, (0.739, 0.948, 0.974)), (0.8, (0.179, 0.702, 0.869))):
        cell = REF.coverage_cell(rho, 3000, 0.05, 1e-3)
        new = E.summarise_coverage(E.coverage_rows(rho, 3000, 0.05, 1e-3, cell["seed"], 0, 300), rho)                    # dieselben ersten 300 Läufe
        for key, o in zip(("iid", "theta_hat", "theta_exact"), old):
            assert abs(new["abdeckung_" + key] - o) < 4 * math.hypot(cell["se_" + key], new["se_" + key]), (rho, key)


def test_erste_fassung_deckt_an_unabhaengigen_daten_zu_oft(monkeypatch):
    monkeypatch.setattr(P, "gpd_cov", _alte_matrix)
    err, se, hit = _pegel_lauf(0.5, 12000, 0.05, 1e-3, 300, 9000, _iid_wait)
    assert hit.mean() > 0.99 and err.std(ddof=1) / se.mean() < 0.6                                                      # richtig: etwa 0,93 und 1,0


def test_erste_fassung_block_schwelle_reproduziert_die_alten_zahlen(monkeypatch):
    import clu_make_coverage as MK
    import clu_reference as REF
    monkeypatch.setattr(P, "gpd_cov", _alte_matrix)
    row = REF.block_schwelle_row(0.8)
    a = np.array(MK.h_chunk((0.8, row["seed"], 0, 100)), dtype=float)
    for col, old in ((3, 0.955), (4, 0.820), (5, 0.491)):                      # k = 50: alte Fassung, bereinigt mit exaktem θ, bereinigt ohne θ
        m = float(np.nanmean(a[:, col]))
        assert abs(m - old) < 4 * math.hypot(math.sqrt(old * (1 - old) / 1500), math.sqrt(max(m * (1 - m), 1e-3) / 100)), (col, m)
