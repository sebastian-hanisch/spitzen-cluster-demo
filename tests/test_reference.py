"""Eingefrorene Messungen (`clu_*.json`): Form, Monotonie, Quelle und Stand je Datei, Rückrechnung der Werkzeuge (Neu-Rechnen als Band, nie auf das letzte Bit)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import clu_constants as C
import clu_evaluation as E
import clu_kernel as K
import clu_queue as Q
import clu_reference as REF

ROOT = Path(__file__).resolve().parent.parent
FILES = ("clu_exact.json", "clu_blocks.json", "clu_coverage.json")


@pytest.mark.parametrize("name", FILES)
def test_jede_datei_nennt_quelle_stand_und_software(name):
    d = json.loads((ROOT / name).read_text(encoding="utf-8"))
    assert "tools/clu_make_" in d["quelle"] and d["stand"] == "2026-11-20"
    assert set(d["software"]) == {"python", "numpy"}


def test_tagesmaximum_verteilungen_sind_monoton_und_decken_alle_zellen_ab():
    for rho in C.RHO2_OPTIONS:
        for n in C.N2_OPTIONS:
            xs, cdf = REF.day_cdf(rho, n)
            assert len(xs) == len(cdf) > 20
            assert np.all(np.diff(cdf) >= -1e-12) and cdf[0] < 0.1 and cdf[-1] >= 1 - 2e-6
            assert np.all(np.diff(xs) == pytest.approx(0.5))


def test_quantile_geordnet_und_wachsen_mit_n_und_last():
    for rho in C.RHO2_OPTIONS:
        for n in C.N2_OPTIONS:
            qs = [REF.day_quantile(rho, n, q) for q in C.Q2_OPTIONS]
            assert all(a < b for a, b in zip(qs, qs[1:]))
    for q in C.Q2_OPTIONS:
        for rho in C.RHO2_OPTIONS:
            col = [REF.day_quantile(rho, n, q) for n in C.N2_OPTIONS]
            assert all(a < b for a, b in zip(col, col[1:]))
        for n in C.N2_OPTIONS:
            row = [REF.day_quantile(rho, n, q) for rho in C.RHO2_OPTIONS]
            assert all(a < b for a, b in zip(row, row[1:]))


def test_quantile_der_pflichtzahlen_gleich_der_geschlossenen_randverteilung_bei_n_gleich_1_naeherung():
    # Gegenprobe mit anderem Rechenweg: das Quantil aus der eingefrorenen Verteilung (Raster 0,5, linear) gegen das Feinraster-Quantil der Tabelle
    for rho, n, q in ((0.8, 256, 0.5), (0.8, 256, 0.99), (0.5, 256, 0.999), (0.9, 256, 0.1)):
        xs, cdf = REF.day_cdf(rho, n)
        assert float(np.interp(q, cdf, xs)) == pytest.approx(REF.day_quantile(rho, n, q), abs=0.06)


def test_frozen_tagesmaximum_gleich_neu_gerechnet():
    g = K.grid_for(0.5, 0.1, 1e-4)
    for n, x in ((16, 3.0), (64, 5.5), (256, 8.0), (1024, 10.5)):
        xs, cdf = REF.day_cdf(0.5, n)
        i = int(round((x - xs[0]) / 0.5))
        assert cdf[i] == pytest.approx(g.day_max_cdf(x, n), abs=1e-6)


def test_theta_u_der_datei_naehert_sich_dem_grenzwert():
    for rho in C.RHO2_OPTIONS:
        vals = [REF.theta_u(rho, fb)["theta"] for fb in (1e-2, 1e-3, 1e-4, 1e-6)]
        assert vals[0] > vals[1] > vals[2] > 0.99 * (1 - rho) ** 2
        assert vals[3] == pytest.approx((1 - rho) ** 2, rel=0.01)
    assert REF.theta_u(0.8, 1e-3)["theta"] == pytest.approx(0.0404, abs=5e-4)
    assert REF.theta_u(0.8, 1e-3)["theta"] == pytest.approx(REF.theta_u(0.8, 1e-3, 0.2)["theta"], rel=2e-3)


def test_blockdaten_xi_faellt_mit_der_blocklaenge_und_bleibt_positiv():
    for rho in C.RHO4_OPTIONS:
        rows = E.block_summary(rho)
        for key in ("xi_ks", "xi_ml"):
            vals = [r[key] for r in rows]
            assert all(a > b > 0 for a, b in zip(vals, vals[1:])) and vals[0] > 0.04
        assert all(r["ks_gev"] < r["ks_gumbel"] for r in rows)
        assert all(r["sigma_ks"] < r["sigma_limit"] for r in rows)


def test_frozen_blockverteilung_gleich_dem_kern_an_einer_stelle():
    d = REF.block_row(0.5, 64)
    g = K.grid_for(0.5, 0.1, 1e-4)
    for i in (3, 10, 20):
        assert g.day_max_cdf(d["x"][i], 64) == pytest.approx(d["cdf"][i], abs=1e-6)


def test_abdeckungszellen_vollstaendig_und_standardfehler_nach_formel():
    cells = REF.coverage()["abdeckung"]
    assert len(cells) == len(C.RHO3_OPTIONS) * len(C.N3_OPTIONS) * len(C.FRAC3_OPTIONS) * len(C.P3_OPTIONS) == 36
    for c in cells:
        for key in ("iid", "theta_hat", "theta_exact"):
            m, se, n = c["abdeckung_" + key], c["se_" + key], c["n_" + key]
            assert se == pytest.approx(math.sqrt(m * (1 - m) / n), rel=1e-9)
        assert c["R"] == 2000 and c["x_wahr"] == pytest.approx(Q.mm1_level(c["rho"], c["p"]))
        assert c["abdeckung_iid"] < c["abdeckung_theta_hat"] <= c["abdeckung_theta_exact"] + 0.02
    for rho, n, frac, p in [(c["rho"], c["n"], c["anteil"], c["p"]) for c in cells][:3]:
        assert REF.coverage_cell(rho, n, frac, p)["rho"] == rho
    with pytest.raises(KeyError):
        REF.coverage_cell(0.7, 3000, 0.05, 1e-3)


def test_frozen_abdeckung_gleich_neu_gerechnet_im_band():
    cell = REF.coverage_cell(0.8, 3000, 0.05, 1e-3)
    new = E.summarise_coverage(E.coverage_rows(0.8, 3000, 0.05, 1e-3, cell["seed"], 0, 300), 0.8)       # dieselben ersten 300 Läufe
    for key in ("iid", "theta_hat", "theta_exact"):
        se = math.hypot(cell["se_" + key], new["se_" + key])
        assert abs(cell["abdeckung_" + key] - new["abdeckung_" + key]) < 4 * se


def test_schaetzer_zeilen_und_block_schwelle_zeilen():
    rows = REF.estimator_rows()
    assert len(rows) == 12 and {r["modell"] for r in rows} == {"maxar", "mm1"}
    for r in rows:
        for k in ("runs_r10", "bloecke_b50", "intervalle"):
            assert r[k]["q25"] <= r[k]["median"] <= r[k]["q75"]
    h = REF.coverage()["block_schwelle"]
    assert [r["phi"] for r in h] == list(C.PHI5_OPTIONS)
    for r in h:
        assert r["p_stern"] >= r["p_alt"] - 1e-15 and r["R"] == 1500
        for k in C.K5_OPTIONS:
            assert r[f"k{k}"]["cluster"] == pytest.approx(k * r["theta"])
    with pytest.raises(KeyError):
        REF.block_schwelle_row(0.3)


def test_block_schwelle_ziel_und_niveau_gleich_der_formel():
    for r in REF.coverage()["block_schwelle"]:
        sb = Q.maxar1_block_scale(r["phi"], 50)
        assert r["x_ziel"] == pytest.approx(sb / -math.log(0.99), rel=1e-12)
        assert r["p_stern"] == pytest.approx(1 - math.exp(-1 / r["x_ziel"]), rel=1e-12)
