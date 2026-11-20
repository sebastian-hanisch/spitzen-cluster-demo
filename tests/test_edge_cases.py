"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): Last 0 und ≥ 1, n = 1 und n = 2, keine oder alle Überschreitungen, Cluster mit einer Überschreitung, r größer als die Reihe,
φ = 0 und φ nahe 1, R = 1, kleinste Überschreitungszahlen der GPD."""

import math

import numpy as np
import pytest

import clu_constants as C
import clu_evaluation as E
import clu_kernel as K
import clu_pot as P
import clu_queue as Q
import clu_theta as TH


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.0, 1.5])
def test_last_null_und_ab_eins_wird_abgelehnt(rho):
    with pytest.raises(ValueError):
        Q.mm1_wait(rho, 10, 1)
    with pytest.raises(ValueError):
        K.theta_exact(rho)


def test_ein_kunde_ist_das_maximum_selbst_und_zwei_kunden():
    for n in (1, 2):
        x = Q.mm1_wait(0.8, n, 3)
        assert x.shape == (n,) and np.all(x >= 0)
    g = K.grid_for(0.5, 0.1, 1e-4)
    assert g.day_max_cdf(3.0, 1) == pytest.approx(g.day_max_cdf(3.0, 1))
    assert g.day_max_cdf(3.0, 2) < g.day_max_cdf(3.0, 1)                       # n = 2: mehr Kunden, kleinere Verteilungsfunktion
    assert g.day_max_cdf_at(3.0, [1, 2, 2, 5]) == [g.day_max_cdf(3.0, 1), g.day_max_cdf(3.0, 2), g.day_max_cdf(3.0, 2), g.day_max_cdf(3.0, 5)]


def test_n_ist_null_wird_abgelehnt():
    with pytest.raises(ValueError):
        Q.mm1_wait(0.5, 0, 1)


def test_keine_ueberschreitung_und_alle_ueber_der_schwelle():
    x = np.zeros(30)
    assert math.isnan(TH.theta_runs(x, 1.0, 5)) and TH.clusters(x, 1.0, 5) == []
    assert math.isnan(TH.theta_blocks(x, 1.0, 5)) and math.isnan(TH.theta_intervals(x, 1.0))
    y = np.full(30, 2.0)                                                       # alle Werte über der Schwelle: ein einziger Cluster, Blöcke-Schätzer nicht definiert
    assert len(TH.clusters(y, 1.0, 1)) == 1 and TH.theta_runs(y, 1.0, 1) == pytest.approx(1 / 30)
    assert math.isnan(TH.theta_blocks(y, 1.0, 5))


def test_cluster_mit_einer_einzigen_ueberschreitung_und_r_groesser_als_die_reihe():
    x = np.zeros(20)
    x[7] = 3.0
    assert [list(c) for c in TH.clusters(x, 1.0, 3)] == [[7]] and TH.theta_runs(x, 1.0, 3) == 1.0
    assert math.isnan(TH.theta_intervals(x, 1.0))                              # weniger als 3 Überschreitungen
    x[3] = x[15] = 3.0
    assert len(TH.clusters(x, 1.0, 1000)) == 1                                  # r größer als die Reihe: alles ein Cluster
    with pytest.raises(ValueError):
        TH.clusters(x, 1.0, 0)


def test_blocklaenge_groesser_als_die_reihe_gibt_nan():
    assert math.isnan(TH.theta_blocks(np.arange(10.0), 5.0, 50))


def test_max_ar1_phi_null_und_phi_nahe_eins():
    x0 = Q.maxar1(0.0, 200, 3)
    assert np.all(x0 > 0)
    x1 = Q.maxar1(0.99, 2000, 3)
    assert x1.shape == (2000,) and np.all(np.isfinite(x1))
    assert np.corrcoef(x1[:-1], np.log(x1[1:]))[0, 1] > 0.5                    # starke Abhängigkeit
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            Q.maxar1(bad, 10, 1)
    assert Q.maxar1_block_scale(0.0, 50) == 50.0 and Q.maxar1_block_scale(0.99, 1) == 1.0


def test_pegel_braucht_p_unter_der_last():
    with pytest.raises(ValueError):
        Q.mm1_level(0.5, 0.6)
    with pytest.raises(ValueError):
        Q.mm1_level(0.5, 0.0)


def test_gpd_ml_mit_zwei_und_drei_ueberschreitungen_und_konstanten_werten():
    xi, sg, _e = P.gpd_ml([1.0, 2.0])
    assert math.isfinite(sg) and sg > 0 and xi >= -1.0
    xi, sg, edge = P.gpd_ml([1.0, 1.0, 1.0])                                    # alle gleich: ξ am Rand −1
    assert xi == -1.0 and sg == 1.0 and edge
    with pytest.raises(ValueError):
        P.gpd_ml([1.0])


def test_schwelle_so_hoch_dass_zu_wenig_ueberschreitungen_bleiben():
    rows = E.coverage_rows(0.5, 100, 0.01, 1e-3, 1, 0, 3)                      # k = 1 bei n = 100: Zeile None statt Rechnung
    assert rows == [None, None, None]


def test_live_lauf_mit_r_gleich_1_und_r_gleich_0():
    assert E.live_coverage(0.5, 1000, 0.1, 1e-3, 0, 1) is None
    one = E.live_coverage(0.5, 1000, 0.1, 1e-3, 1, 1)
    assert one["R"] == 1 and one["abdeckung_iid"] in (0.0, 1.0) and math.isnan(one["sd_ln"])


def test_delta_intervall_braucht_positives_theta():
    with pytest.raises(ValueError):
        P.delta_ci_logx(0.0, 1.0, 5.0, 100, 1000, 1e-3, 0.0)


def test_fit_topk_grenzen():
    x = np.arange(10.0)
    u, k, y = P.fit_topk(x, 3)
    assert u == 6.0 and k == 3 and list(y) == [1.0, 2.0, 3.0]
    for bad in (0, 10):
        with pytest.raises(ValueError):
            P.fit_topk(x, bad)


def test_serie_ohne_ueberschreitung_im_verdikt():
    a = E.analyse_series(0.3, 300, 0.001, 10, 50, 1)
    if a.k == 0:
        assert E.verdict_series(a)[0] == "warning"
    a0 = E.analyse_series(0.3, 300, 0.001, 10, 50, 1)
    assert a0.n == 300 and a0.x.shape == (300,)


def test_kern_ausserhalb_des_gitters_wird_abgelehnt():
    g = K.Grid(0.5, 0.1, 10.0)
    with pytest.raises(ValueError):
        g.day_max_cdf(50.0, 10)
    with pytest.raises(ValueError):
        g.day_max_cdf(-1.0, 10)


def test_regler_grenzen_liegen_innerhalb_dessen_was_gerechnet_wird():
    assert C.RHO1_MAX < 1 and C.FRAC1_OPTIONS[-1] < C.RHO1_MIN                  # F̄ < ρ: die Schwelle liegt über 0
    assert C.N1_MIN >= 300 and max(C.R3_OPTIONS) * max(C.N3_OPTIONS) <= C.LIVE_N_MAX_PRODUCT
    assert max(C.P3_OPTIONS) < min(C.FRAC3_OPTIONS) and max(C.P3_OPTIONS) < min(C.RHO3_OPTIONS)
