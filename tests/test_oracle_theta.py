"""Orakel für den Extremalindex und die exakte Verteilung des Tagesmaximums (ORACLE-PLAYBOOK §5, §6): jedes Mal ein anderer Rechenweg als der der Demo.

(a) θ zweifach: die Zerlegungsformel in Brüchen gegen den Hauptwert des abgetöteten Kerns, den `numpy.linalg.eigvals` (voller Eigenwertlöser) liefert, nicht das Potenzverfahren der Demo; zwei Gitterweiten.
(b) Der Kern gegen die Simulation der Definition (stationäre Verteilung, Tagesmaximum) im Band von vier Standardfehlern.
(c) Die exakte Blockverteilung des max-AR(1) gegen Simulation (Kolmogorov-Abstand gegen die kritische Grenze).
(d) Schätzer von θ an exakten Werten (max-AR(1): θ = 1 − φ), Zufall als Band.
(e) Pegel x_p = ln(ρ/p)/γ gegen die Randverteilung einer langen Reihe.
Keine Aussage über Quellen: θ = (1−ρ)² ist eine eigene Herleitung (README, Abschnitt Quellen), die Quelle Peshkova (2026) ist nur auf Kurzseiten-Stufe bekannt und hier NICHT als Test benutzt."""

import math
from fractions import Fraction

import numpy as np
import pytest

import clu_kernel as K
import clu_queue as Q
import clu_theta as TH


def _principal_by_eigvals(grid: K.Grid, j: int) -> float:
    return float(np.max(np.real(np.linalg.eigvals(grid.T[: j + 1, : j + 1]))))


# --- (a) ----------------------------------------------------------------------------------------------------------------------------------
def test_theta_formel_in_bruechen_gleich_kern_per_eigenwertloeser_zwei_gitterweiten():
    rho, f_bar = 0.5, 1e-3
    exact = Fraction(1, 4)
    u = Q.mm1_threshold(rho, f_bar)
    vals = []
    for h in (0.2, 0.1):
        g = K.grid_for(rho, h, f_bar)
        j = g.index(u)
        th_eig = (1.0 - _principal_by_eigvals(g, j)) / float(g.surv[j])
        assert th_eig == pytest.approx(g.theta_at(u), rel=1e-6)                    # Potenzverfahren der Demo gegen vollen Eigenwertlöser
        assert th_eig == pytest.approx(float(exact), rel=0.01)                      # bei F̄ = 10⁻³ liegt θ(u) rund 0,6 % über dem Grenzwert
        assert th_eig > float(exact)
        vals.append(th_eig)
    assert vals[0] == pytest.approx(vals[1], rel=2e-3)                              # Gitterverfeinerung ändert θ(u) um höchstens 0,2 %


@pytest.mark.parametrize("rho", [0.8, 0.9])
def test_theta_u_naehert_sich_von_oben_dem_grenzwert(rho):
    th = [K.theta_of_threshold(rho, fb, 0.2) for fb in (1e-2, 1e-3, 1e-4)]
    exact = (1 - rho) ** 2
    assert th[0] > th[1] > th[2] > exact * 0.999                                    # fällt mit sinkendem F̄ auf (1−ρ)² zu
    assert th[2] == pytest.approx(exact, rel=0.01)


def test_frozen_theta_u_gleich_neu_gerechnet_im_band():
    import clu_reference as REF
    for rho, fb in ((0.5, 1e-2), (0.5, 1e-4), (0.8, 1e-3)):
        assert K.theta_of_threshold(rho, fb, 0.2) == pytest.approx(REF.theta_u(rho, fb, 0.2)["theta"], rel=1e-5)   # anderes Gitterende (Rest oberhalb): Abstand rund 5e-7


# --- (b) ----------------------------------------------------------------------------------------------------------------------------------
def test_stationaere_verteilung_der_simulation_gleich_der_geschlossenen_randverteilung():
    rho = 0.8
    x = Q.mm1_wait(rho, 2_000_000, 3)[::500]                                        # jeder 500. Kunde: nahezu unabhängig (Korrelationszeit rund 25 Kunden)
    for xp in (2.0, 8.0, 14.0):
        frac = float(np.mean(x > xp))
        target = float(Q.mm1_sf(rho, xp))
        assert abs(frac - target) < 6 * math.sqrt(target * (1 - target) / len(x))   # Faktor 6 statt 4: die dünnen Werte sind nur nahezu unabhängig
    assert abs(float(np.mean(x > 0)) - rho) < 6 * math.sqrt(rho * (1 - rho) / len(x))


@pytest.mark.parametrize("rho,n,x", [(0.8, 16, 6.0), (0.8, 64, 12.0), (0.5, 256, 7.8)])
def test_tagesmaximum_kern_gleich_simulation_der_definition(rho, n, x):
    g = K.grid_for(rho, 0.1, 1e-4)
    exact = g.day_max_cdf(x, n)
    runs = 3000
    hits = sum(Q.mm1_wait(rho, n, 4242, r).max() <= x for r in range(runs))
    se = math.sqrt(max(exact * (1 - exact), 1e-6) / runs)
    assert abs(hits / runs - exact) < 4 * se


# --- (c) ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_max_ar1_blockverteilung_exakt_kolmogorov_abstand(phi):
    runs, b = 10000, 50
    m = np.sort([Q.maxar1(phi, b, 99, r).max() for r in range(runs)])
    cdf = Q.maxar1_block_cdf(phi, b, m)
    ks = float(np.max(np.maximum(np.abs(cdf - np.arange(1, runs + 1) / runs), np.abs(cdf - np.arange(0, runs) / runs))))
    assert ks < 1.628 / math.sqrt(runs)                                             # kritische Grenze zum Niveau 1 %


def test_max_ar1_rand_ist_frechet_eins():
    x = np.concatenate([Q.maxar1(0.8, 200, 5, r) for r in range(200)])[::7]          # dünn: nahezu unabhängig
    for q in (0.5, 0.9, 0.99):
        lev = -1.0 / math.log(q)
        assert abs(float(np.mean(x <= lev)) - q) < 4 * math.sqrt(q * (1 - q) / len(x)) * 2


# --- (d) ----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("phi,rel_blocks", [(0.5, 0.10), (0.8, 0.15)])
def test_schaetzer_an_exakten_werten_median_im_band(phi, rel_blocks):
    u = -1.0 / math.log(0.95)
    est = {"runs": [], "blocks": [], "int": []}
    for r in range(40):
        x = Q.maxar1(phi, 5000, 31, r)
        est["runs"].append(TH.theta_runs(x, u, 10))
        est["blocks"].append(TH.theta_blocks(x, u, 50))
        est["int"].append(TH.theta_intervals(x, u))
    theta = 1 - phi
    assert np.median(est["blocks"]) == pytest.approx(theta, rel=rel_blocks)
    assert np.median(est["int"]) == pytest.approx(theta, rel=rel_blocks)
    assert np.median(est["runs"]) < theta                                           # Runs mit festem r = 10 liegt bei 5 % über der Schwelle unter dem exakten Wert


def test_runs_schaetzer_mit_festem_r_ist_bei_unabhaengigen_daten_verzerrt():
    u = -1.0 / math.log(0.95)
    runs = [TH.theta_runs(Q.maxar1(0.0, 5000, 17, r), u, 10) for r in range(60)]
    blocks = [TH.theta_blocks(Q.maxar1(0.0, 5000, 17, r), u, 50) for r in range(60)]
    assert np.median(runs) == pytest.approx(0.60, abs=0.03)                         # nicht 1: Abstände unter 11 sind bei 5 % Überschreitungen häufig
    assert np.median(blocks) == pytest.approx(1.0, abs=0.05)


# --- (e) ----------------------------------------------------------------------------------------------------------------------------------
def test_pegel_formel_gleich_randverteilung_und_exakte_werte():
    assert Q.mm1_level(0.8, 1e-3) == pytest.approx(math.log(800) / 0.2, rel=1e-12)
    assert Q.mm1_level(0.5, 1e-3) == pytest.approx(math.log(500) / 0.5, rel=1e-12)
    assert float(Q.mm1_sf(0.8, Q.mm1_level(0.8, 1e-3))) == pytest.approx(1e-3, rel=1e-9)
    assert Q.mm1_level(0.8, 1e-3) == pytest.approx(33.4, abs=0.05) and Q.mm1_level(0.5, 1e-3) == pytest.approx(12.4, abs=0.05)
