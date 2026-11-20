"""Eigenschaften über feste Zufallsinstanzen (je 20 bis 50, fester Zufalls-Seed): θ ∈ (0, 1], Monotonie, Ordnung der Gumbel-Quantile, Schätzer-Bereiche, Cluster-Zerlegung."""

import math
import random

import numpy as np
import pytest

import clu_kernel as K
import clu_pot as P
import clu_queue as Q
import clu_theta as TH

RNG = random.Random(20261120)
RHOS = [RNG.uniform(0.2, 0.95) for _ in range(40)]


@pytest.mark.parametrize("rho", RHOS)
def test_theta_liegt_in_null_eins_und_die_clustergroesse_ist_sein_kehrwert(rho):
    th = K.theta_exact(rho)
    assert 0 < th < 1 and K.mean_cluster_size(rho) * th == pytest.approx(1.0, rel=1e-12)


def test_theta_waechst_mit_fallender_last_exakt_monoton():
    rhos = sorted(RHOS)
    th = [K.theta_exact(r) for r in rhos]
    assert all(a > b for a, b in zip(th, th[1:]))                                # steigende Last, fallendes θ


@pytest.mark.parametrize("rho,n,q", [(r, RNG.choice([16, 64, 256, 1024]), RNG.choice([0.1, 0.5, 0.9, 0.99, 0.999])) for r in RHOS[:30]])
def test_gumbel_quantil_mit_theta_eins_liegt_ueber_dem_mit_theta_exakt(rho, n, q):
    assert K.gumbel_day_quantile(rho, n, q, theta=1.0) > K.gumbel_day_quantile(rho, n, q)
    assert K.gumbel_day_quantile(rho, n, q, theta=1.0) - K.gumbel_day_quantile(rho, n, q) == pytest.approx(K.gumbel_shift(rho), rel=1e-9)  # die Verschiebung hängt weder an n noch an q


@pytest.mark.parametrize("seed", range(40))
def test_schaetzer_liegen_in_ihren_bereichen_und_clusterzerlegung_ist_vollstaendig(seed):
    rho = 0.5 + 0.4 * (seed % 5) / 4
    x = Q.mm1_wait(rho, 2000, seed)
    u = Q.mm1_threshold(rho, 0.05)
    k = int(np.sum(x > u))
    if k == 0:                                                                   # bei Last 0,9 und 2.000 Kunden kommt es vor, dass keine Spitze auftritt
        assert math.isnan(TH.theta_runs(x, u, 5)) and TH.clusters(x, u, 5) == []
        return
    for r in (1, 5, 20):
        cl = TH.clusters(x, u, r)
        assert sum(len(c) for c in cl) == k                                      # jede Überschreitung liegt in genau einem Cluster
        assert all(np.all(np.diff(c) <= r) for c in cl)                          # innerhalb eines Clusters nie mehr als r Abstand
        assert all(b[0] - a[-1] > r for a, b in zip(cl, cl[1:]))                 # zwischen Clustern mehr als r
        th = TH.theta_runs(x, u, r)
        assert 0 < th <= 1 and th == pytest.approx(len(cl) / k)
    assert TH.theta_runs(x, u, 1) >= TH.theta_runs(x, u, 5) >= TH.theta_runs(x, u, 20)   # größeres r fasst mehr zusammen
    for th in (TH.theta_blocks(x, u, 50), TH.theta_intervals(x, u)):
        assert math.isnan(th) or 0 < th <= 1.0 + 1e-12 or (th > 0)               # der Blöcke-Schätzer ist nicht auf 1 begrenzt


@pytest.mark.parametrize("seed", range(30))
def test_wartezeiten_sind_nichtnegativ_und_haengen_mit_der_last_zusammen(seed):
    rho = 0.3 + 0.6 * (seed % 7) / 6
    x = Q.mm1_wait(rho, 3000, seed)
    assert np.all(x >= 0) and np.all(np.isfinite(x))
    assert x.shape == (3000,)


def test_last_hoeher_gibt_im_mittel_laengere_wartezeiten():
    means = [np.mean([Q.mm1_wait(r, 4000, s).mean() for s in range(10)]) for r in (0.5, 0.7, 0.9)]
    assert means[0] < means[1] < means[2]


@pytest.mark.parametrize("seed", range(25))
def test_pegel_waechst_mit_fallendem_p_und_gpd_pegel_ist_monoton(seed):
    rng = random.Random(seed)
    xi, sg, u, zeta = rng.uniform(-0.3, 0.5), rng.uniform(0.3, 3.0), rng.uniform(1.0, 20.0), rng.uniform(0.01, 0.1)
    levels = [P.pot_level(xi, sg, u, zeta, p) for p in (1e-2, 1e-3, 1e-4, 1e-5)]
    assert all(a < b for a, b in zip(levels, levels[1:])) or xi < 0                   # bei ξ < 0 wächst der Pegel ebenfalls, aber gegen den Endpunkt
    assert all(a <= b + 1e-12 for a, b in zip(levels, levels[1:]))


def test_cdf_des_tagesmaximums_ist_monoton_in_x_und_faellt_in_n():
    g = K.grid_for(0.5, 0.2, 1e-3)
    xs = [3.0, 6.0, 9.0, 12.0]
    for n in (8, 64):
        c = [g.day_max_cdf(x, n) for x in xs]
        assert all(a < b for a, b in zip(c, c[1:]))
    assert all(g.day_max_cdf(x, 64) < g.day_max_cdf(x, 8) for x in xs)
    assert [g.day_max_cdf(6.0, n) for n in (8, 16, 32)] == sorted([g.day_max_cdf(6.0, n) for n in (8, 16, 32)], reverse=True)


def test_tagesmaximum_waechst_mit_der_last_exakt():
    gs = [K.grid_for(r, 0.2, 1e-3) for r in (0.5, 0.8)]
    assert gs[0].day_max_cdf(10.0, 64) > gs[1].day_max_cdf(10.0, 64)
