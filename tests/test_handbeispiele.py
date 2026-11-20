"""Die Handbeispiele des Plans als exakte Tests (Brüche, Ganzzahlen, Tafelwerte) und je Beispiel ein zweiter Rechenweg.

H1 θ = (1−ρ)² in Brüchen; H2 das diskrete Gegenstück θ = 1/6; H3 die Schätzer an 20 Beobachtungen von Hand; H4 max-AR(1) mit φ = 1/2; H5 der Kern gegen die geschlossene Randverteilung;
H6 Cluster unter den Überschreitungen; H7 Block-Ziel und Niveau je Beobachtung. Zufall nur aus der kopierten Vorlage (`clu_rng`), Zufallsanteile als Band mit Standardfehler."""

import math
from fractions import Fraction

import numpy as np
import pytest

import clu_kernel as K
import clu_pot as P
import clu_queue as Q
import clu_rng
import clu_theta as TH


# --- H1 -----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [Fraction(1, 2), Fraction(4, 5), Fraction(9, 10)])
def test_h1_theta_in_bruechen(rho):
    g = 1 - rho
    for s in (Fraction(1, 3), Fraction(2), Fraction(7, 5)):                  # Laplace-Transformierte der Verweilzeit T = S + W ist die einer Exp(γ)
        assert ((1 - rho) + rho * g / (g + s)) / (1 + s) == g / (g + s)
    p_t_lt_a = g / (g + rho)                                                  # P(T < A) für T ~ Exp(γ), A ~ Exp(ρ)
    e = rho * g / (rho + g) ** 2                                              # E[e^{γ(T−A)}; T < A]
    assert g + rho == 1 and p_t_lt_a == g and e == rho * g
    assert p_t_lt_a - e == (1 - rho) ** 2
    assert K.theta_exact(float(rho)) == pytest.approx(float((1 - rho) ** 2), rel=1e-12)


def test_h1_mittlere_clustergroesse_4_25_100():
    assert [1 / (1 - Fraction(a, b)) ** 2 for a, b in ((1, 2), (4, 5), (9, 10))] == [4, 25, 100]
    assert [K.mean_cluster_size(r) for r in (0.5, 0.8, 0.9)] == pytest.approx([4, 25, 100], rel=1e-9)


def test_h1_verschiebung_grenzwert_und_gumbel_median():
    shift = math.log(25) / 0.2
    assert shift == pytest.approx(16.09, abs=0.01) and 3 * shift == pytest.approx(48.3, abs=0.1)
    assert K.gumbel_shift(0.8) == pytest.approx(shift, rel=1e-9)
    assert K.gumbel_day_quantile(0.8, 256, 0.5) == pytest.approx(12.35, abs=0.01)
    assert K.gumbel_day_quantile(0.8, 256, 0.5, theta=1.0) == pytest.approx(28.44, abs=0.01)


# --- H2 -----------------------------------------------------------------------------------------------------------------------------------
def _walk_kernel(a, b, u):
    Qm = np.zeros((u + 1, u + 1))
    for i in range(u + 1):
        Qm[i, max(i - 1, 0)] += float(b)
        if i + 1 <= u:
            Qm[i, i + 1] += float(a)
    return Qm


def test_h2_diskretes_gegenstueck_ein_sechstel():
    a, b = Fraction(1, 3), Fraction(2, 3)
    r = a / b
    assert r == Fraction(1, 2) and b * (1 - r) ** 2 == Fraction(1, 6)
    for u, tol in ((6, 0.0090), (10, 9e-4), (14, 8e-5), (20, 3e-6)):          # Toleranz = tatsächlicher Abstand, von oben
        lam = float(np.max(np.real(np.linalg.eigvals(_walk_kernel(a, b, u)))))
        th_u = (1 - lam) / float(r) ** (u + 1)
        assert 0 < th_u - 1 / 6 < tol


def _cluster_prob(u, R, num):
    a, b = Fraction(1, 3), Fraction(2, 3)
    r = a / b
    v = [num(0)] * (u + 1)
    v[u] = num(1 - r) * num(b)
    for _ in range(R - 1):
        w = [num(0)] * (u + 1)
        for i, p_ in enumerate(v):
            if p_ == 0:
                continue
            w[max(i - 1, 0)] += p_ * num(b)
            if i + 1 <= u:
                w[i + 1] += p_ * num(a)
        v = w
    return sum(v)


def test_h2_clusterwahrscheinlichkeit_exakter_bruch_matrixpotenz_und_simulation():
    cp = _cluster_prob(5, 40, Fraction)
    assert isinstance(cp, Fraction) and abs(float(cp) - 0.159186648431680) < 1e-12
    Qm = _walk_kernel(Fraction(1, 3), Fraction(2, 3), 5)
    v5 = np.zeros(6)
    v5[5] = 0.5 * (2 / 3)
    assert float((v5 @ np.linalg.matrix_power(Qm, 39)).sum()) == pytest.approx(float(cp), abs=1e-12)
    rng = clu_rng.SplitMix64(311)
    runs, hits = 20000, 0
    for _ in range(runs):
        j = 1
        while rng.random() < 0.5:
            j += 1
        w, ok = 5 + j, True
        for _s in range(40):
            w = max(w + (1 if rng.random() < 1 / 3 else -1), 0)
            if w > 5:
                ok = False
                break
        hits += ok
    se = math.sqrt(float(cp) * (1 - float(cp)) / runs)
    assert abs(hits / runs - float(cp)) < 4 * se


# --- H3 -----------------------------------------------------------------------------------------------------------------------------------
def test_h3_schaetzer_von_hand(mini):
    x, u = mini
    idx = TH.exceedance_positions(x, u)
    assert list(np.diff(idx)) == [1, 1, 7, 1, 6]
    assert len(TH.clusters(x, u, 2)) == 3
    assert TH.theta_runs(x, u, 2) == 0.5                                      # 3 Cluster / 6 Überschreitungen
    assert TH.theta_blocks(x, u, 5) == pytest.approx(math.log(0.25) / (5 * math.log(0.7)), rel=1e-12)
    assert TH.theta_blocks(x, u, 5) == pytest.approx(0.7774, abs=5e-4)


def test_h3_intervall_schaetzer_von_hand(mini):
    x, u = mini
    # N = 6 Überschreitungen, 5 Abstände T = 1, 1, 7, 1, 6; größter Abstand > 2: θ = 2 (Σ(T−1))² / ((N−1) Σ(T−1)(T−2)) = 2·11² / (5·50)
    assert TH.theta_intervals(x, u) == pytest.approx(min(1.0, 2 * 121 / (5 * 50)))
    assert TH.theta_intervals(x, u) == pytest.approx(0.968)


# --- H4 -----------------------------------------------------------------------------------------------------------------------------------
def test_h4_max_ar1_exponent_und_median():
    phi = Fraction(1, 2)
    assert [1 + (m - 1) * (1 - phi) for m in (1, 2, 3, 10)] == [1, Fraction(3, 2), 2, Fraction(11, 2)]
    assert Q.maxar1_block_scale(0.5, 3) == 2.0
    x_med = 2 / math.log(2)
    assert x_med == pytest.approx(2.885, abs=1e-3) and 3 / math.log(2) == pytest.approx(4.328, abs=1e-3)
    assert float(Q.maxar1_block_cdf(0.5, 3, x_med)) == pytest.approx(0.5, abs=1e-12)


def test_h4_max_ar1_median_in_20000_laeufen_im_band():
    runs, hits = 20000, 0
    x_med = 2 / math.log(2)
    for r in range(runs):
        x = Q.maxar1(0.5, 3, 707, r)
        hits += x.max() <= x_med
    assert abs(hits / runs - 0.5) < 4 * math.sqrt(0.25 / runs)


# --- H5 -----------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grid05():
    return K.Grid(0.5, 0.1, 40.0)


def test_h5_stationaer_und_randverteilung_gegen_geschlossene_formel(grid05):
    assert grid05.pi[0] == pytest.approx(0.5, abs=5e-3)
    assert grid05.surv[30] == pytest.approx(0.5 * math.exp(-0.5 * 3.0), abs=3e-3)
    assert grid05.day_max_cdf(4.0, 1) == pytest.approx(1 - 0.5 * math.exp(-0.5 * 4.0), abs=3e-3)


def test_h5_tagesmaximum_n256_und_quantil(grid05):
    assert grid05.day_max_cdf(7.8, 256) == pytest.approx(0.50267, abs=2e-3)
    xs = [5.3, 5.4, 5.5, 5.6, 5.7]
    cs = [grid05.day_max_cdf(x, 256) for x in xs]
    assert cs[2] < 0.1 < cs[3]                                                # die frühere Zahl 5,4 war eine Gitterstelle daneben
    assert 5.45 < float(np.interp(0.1, cs, xs)) < 5.60


def test_h5_gitterverfeinerung_und_simulation_n64():
    g1, g2 = K.Grid(0.5, 0.1, 40.0), K.Grid(0.5, 0.05, 40.0)
    c1, c2 = g1.day_max_cdf(6.0, 64), g2.day_max_cdf(6.0, 64)
    assert c2 == pytest.approx(c1, abs=2e-3)
    runs = 3000
    hits = sum(Q.mm1_wait(0.5, 64, 909, r).max() <= 6.0 for r in range(runs))
    se = math.sqrt(c1 * (1 - c1) / runs)
    assert abs(hits / runs - c1) < 4 * se


# --- H6 -----------------------------------------------------------------------------------------------------------------------------------
def test_h6_cluster_unter_den_ueberschreitungen():
    assert 150 * Fraction(1, 25) == 6 and 150 * Fraction(1, 4) == Fraction(75, 2)
    assert K.clusters_among(150, 0.04) == pytest.approx(6.0) and K.clusters_among(150, 0.25) == pytest.approx(37.5)
    assert 1 / math.sqrt(0.25) == 2.0 and 1 / math.sqrt(0.04) == 5.0


def test_h6_breite_waechst_mit_eins_durch_wurzel_theta():
    xi, sg, u, k, n, p = 0.0, 1.0, 5.0, 150, 3000, 1e-3
    lo1, hi1 = P.delta_ci_logx(xi, sg, u, k, n, p, 1.0)
    for th in (0.25, 0.04):
        lo, hi = P.delta_ci_logx(xi, sg, u, k, n, p, th)
        assert (hi - lo) / (hi1 - lo1) == pytest.approx(1 / math.sqrt(th), rel=1e-9)


# --- H7 -----------------------------------------------------------------------------------------------------------------------------------
def test_h7_block_ziel_und_niveau_je_beobachtung():
    phi8, phi5 = Fraction(4, 5), Fraction(1, 2)
    assert 1 + (2 - 1) * (1 - phi8) == Fraction(6, 5) and 1 + 49 * (1 - phi8) == Fraction(54, 5) and 1 + 49 * (1 - phi5) == Fraction(51, 2)
    xs8 = 10.8 / -math.log(0.99)
    assert math.exp(-10.8 / xs8) == pytest.approx(0.99, abs=1e-12) and xs8 == pytest.approx(1074.59, abs=0.01)
    p_obs = 1 - 0.99 ** (1 / 50)
    pstar8 = 1 - math.exp(-1 / xs8)
    pstar5 = 1 - math.exp(-1 / (25.5 / -math.log(0.99)))
    assert p_obs == pytest.approx(2.0099e-4, abs=2e-8) and pstar8 == pytest.approx(9.302e-4, abs=2e-7)
    assert pstar8 / p_obs == pytest.approx(4.63, abs=0.02) and pstar5 / p_obs == pytest.approx(1.96, abs=0.02)
    assert Q.frechet_level(p_obs) == pytest.approx(4975.0, abs=0.1) and Q.frechet_level(p_obs) / xs8 == pytest.approx(4.63, abs=0.02)
    assert Q.maxar1_block_scale(0.8, 50) == pytest.approx(10.8) and Q.maxar1_block_scale(0.5, 50) == pytest.approx(25.5)
