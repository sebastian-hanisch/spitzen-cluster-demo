"""Claims: jede Zahl der README gegen die exakte Rechnung oder die eingefrorene Messung (Rundungsband: halbe Einheit der letzten gedruckten Stelle).

`claim("0,2612", wert)` prüft zweierlei: die Zahl steht in der README, und der Wert, auf die gedruckten Stellen gerundet, ist die gedruckte Zahl. „Messung“ ist die Messung der eingefrorenen Dateien
(Zähleinheit Läufe, `clu_*.json`); ihre Neu-Rechnung als Band steht in `test_reference.py`, exakte Zahlen stehen in den Handbeispielen und im Orakel."""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import clu_constants as C
import clu_evaluation as E
import clu_kernel as K
import clu_queue as Q
import clu_reference as REF
from test_handbeispiele import _cluster_prob

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def claim(printed, value, tol_extra=0.0):
    assert printed in README, f"„{printed}“ steht nicht in der README"
    txt = printed.replace("−", "-").replace("+", "").replace(" ", "")
    txt = re.sub(r"[^0-9,\-]", "", txt.replace(".", "")) if "," in txt or "." in txt else re.sub(r"[^0-9\-]", "", txt)
    decimals = len(txt.split(",")[1]) if "," in txt else 0
    got = float(txt.replace(",", "."))
    assert abs(value - got) <= 0.5 * 10 ** (-decimals) + 1e-9 + tol_extra, (printed, value)


def cell(rho, n=3000, frac=0.05, p=1e-3):
    return REF.coverage_cell(rho, n, frac, p)


def pc(x):
    return 100 * x


# --- θ und der Kern ---------------------------------------------------------------------------------------------------------------------------
def test_theta_brueche_und_clustergroessen():
    for printed in ("1/4", "1/25", "1/100"):
        assert printed in README
    assert [K.theta_exact(r) for r in (0.5, 0.8, 0.9)] == pytest.approx([1 / 4, 1 / 25, 1 / 100])
    assert "**4, 25, 100 Überschreitungen je Cluster**" in README and [K.mean_cluster_size(r) for r in (0.5, 0.8, 0.9)] == pytest.approx([4, 25, 100])


def test_theta_u_tabelle():
    rows = {0.5: ("0,2612", "0,2516", "0,2501", "0,2499"), 0.8: ("0,0427", "0,0404", "0,0400", "0,0400"), 0.9: ("0,0107", "0,0101", "0,0100", "0,0100")}
    for rho, printed in rows.items():
        for fb, s in zip((1e-2, 1e-3, 1e-4, 1e-6), printed):
            claim(s, REF.theta_u(rho, fb)["theta"])
    rel = [abs(REF.theta_u(r, fb, 0.2)["theta"] / REF.theta_u(r, fb)["theta"] - 1) for r in (0.5, 0.8, 0.9) for fb in (1e-2, 1e-3, 1e-4, 1e-6)]
    assert max(rel) <= 0.003 and "höchstens 0,3 %" in README
    assert "rund 1 % über dem Grenzwert" in README
    over3 = [REF.theta_u(r, 1e-3)["theta"] / (1 - r) ** 2 - 1 for r in (0.5, 0.8, 0.9)]
    over2 = [REF.theta_u(r, 1e-2)["theta"] / (1 - r) ** 2 - 1 for r in (0.5, 0.8, 0.9)]
    assert all(0.005 <= v <= 0.012 for v in over3) and 0.04 <= min(over2) and max(over2) <= 0.08 and "4 bis 8 %" in README


def test_theta_u_bei_zu_knappem_gitterrand_ist_verfaelscht():
    u = Q.mm1_threshold(0.8, 1e-3)
    ref = K.Grid(0.8, 0.2, u + 60).theta_at(u)
    assert K.Grid(0.8, 0.2, u + 6).theta_at(u) / ref - 1 == pytest.approx(0.32, abs=0.01)
    u9 = Q.mm1_threshold(0.9, 1e-3)
    assert K.Grid(0.9, 0.2, u9 + 6).theta_at(u9) / K.Grid(0.9, 0.2, u9 + 60).theta_at(u9) - 1 == pytest.approx(0.99, abs=0.02)
    assert "um 32 % (Last 0,8) bis 99 % (Last 0,9)" in README
    claim("0,0100", REF.theta_u(0.9, 1e-6)["theta"])
    claim("0,2499", REF.theta_u(0.5, 1e-6)["theta"])


def test_diskretes_gegenstueck():
    assert "θ = 1/6" in README and "0,159187" in README
    claim("0,159187", float(_cluster_prob(5, 40, Fraction)))


# --- Tagesmaximum -----------------------------------------------------------------------------------------------------------------------------
def test_tagesmaximum_quantile_tabelle():
    rows = {0.5: ("5,5", "7,8", "16,2", "20,8"), 0.8: ("9,2", "14,0", "34,3", "45,8"), 0.9: ("11,9", "20,0", "59,5", "82,6")}
    for rho, printed in rows.items():
        for q, s in zip((0.1, 0.5, 0.99, 0.999), printed):
            assert s in README
            assert abs(REF.day_quantile(rho, 256, q) - float(s.replace(",", "."))) <= 0.05 + 1e-9


def test_unabhaengig_gerechnet_und_verschiebung():
    for q, s in zip((0.1, 0.5, 0.99, 0.999), ("22,4", "28,4", "49,6", "61,1")):
        claim(s, K.gumbel_day_quantile(0.8, 256, q, theta=1.0))
    shifts = [K.gumbel_day_quantile(0.8, 256, q, theta=1.0) - REF.day_quantile(0.8, 256, q) for q in (0.1, 0.5, 0.99, 0.999)]
    for s, v in zip(("13,3", "14,5", "15,3", "15,4"), shifts):
        claim(s, v)
    mins = [3 * v for v in shifts[1:]]
    assert round(min(mins)) == 43 and round(max(mins)) == 46 and "**43 bis 46 min**" in README
    claim("16,1", K.gumbel_shift(0.8))
    assert round(3 * K.gumbel_shift(0.8)) == 48 and "(48 min)" in README
    assert "14,4 (28,4 − 14,0 aus gerundeten Zahlen)" in README


def test_gumbel_mit_theta_rand_und_koerper():
    for q, s, e in ((0.99, "33,5", "34,3"), (0.999, "45,1", "45,8")):
        claim(s, K.gumbel_day_quantile(0.8, 256, q))
        claim(e, REF.day_quantile(0.8, 256, q))
    claim("−2,2", pc(K.gumbel_day_quantile(0.8, 256, 0.99) / REF.day_quantile(0.8, 256, 0.99) - 1))
    claim("−1,6", pc(K.gumbel_day_quantile(0.8, 256, 0.999) / REF.day_quantile(0.8, 256, 0.999) - 1))
    claim("6,3", K.gumbel_day_quantile(0.8, 256, 0.1))
    claim("9,2", REF.day_quantile(0.8, 256, 0.1))
    claim("12,0", K.gumbel_day_quantile(0.9, 256, 0.5))
    claim("20,0", REF.day_quantile(0.9, 256, 0.5))
    assert abs(K.gumbel_day_quantile(0.9, 256, 0.1)) < 0.05 and "0,1-Quantil 0,0 gegen 11,9" in README
    claim("11,9", REF.day_quantile(0.9, 256, 0.1))
    claim("−8,7", pc(K.gumbel_day_quantile(0.9, 256, 0.99) / REF.day_quantile(0.9, 256, 0.99) - 1))
    claim("−6,2", pc(K.gumbel_day_quantile(0.9, 256, 0.999) / REF.day_quantile(0.9, 256, 0.999) - 1))
    claim("8,2", 256 * K.theta_exact(0.8) * 0.8)
    claim("2,3", 256 * K.theta_exact(0.9) * 0.9)


def test_exakte_quantile_sind_nicht_die_der_langen_folge():
    assert "14,0 / 36,1 / 49,0" in README and "14,0 / 34,3 / 45,8" in README
    for q, v in zip((0.5, 0.99, 0.999), (14.0, 34.3, 45.8)):
        assert abs(REF.day_quantile(0.8, 256, q) - v) <= 0.05
    assert README.count("36,1") == 1


# --- Populations-ξ ----------------------------------------------------------------------------------------------------------------------------
def test_populations_xi():
    rows = E.block_summary(0.8)
    for s, r in zip(("+0,159", "+0,087", "+0,043", "+0,018"), rows):
        claim(s, r["xi_ks"])
    for s, r in zip(("+0,151", "+0,083", "+0,040", "+0,016"), rows):
        claim(s, r["xi_ml"])
    for s, r in zip(("0,0216", "0,0120", "0,0059", "0,0025"), rows):
        claim(s, r["ks_gumbel"])
    for s, r in zip(("0,0055", "0,0029", "0,0016", "0,0007"), rows):
        claim(s, r["ks_gev"])
    for s, r in zip(("3,56", "4,12", "4,56", "4,82"), rows):
        claim(s, r["sigma_ks"])
    for s, r in zip(("+0,057", "+0,025", "+0,010", "+0,003"), E.block_summary(0.5)):
        claim(s, r["xi_ks"])
    assert "Grenzwert 5,0" in README and rows[0]["sigma_limit"] == pytest.approx(5.0)
    assert "0,0050 / 0,0028 / 0,0016 / 0,0007" in README                                               # Vormessung: nur als Vergleich zitiert


# --- Abdeckung --------------------------------------------------------------------------------------------------------------------------------
def test_abdeckung_last_50_und_80():
    c5, c8 = cell(0.5), cell(0.8)
    for s, key in (("35,4", "iid"), ("59,1", "theta_hat"), ("61,7", "theta_exact")):
        claim(s, pc(c5["abdeckung_" + key]))
    for s, key in (("1,1", "iid"), ("1,1", "theta_hat"), ("1,1", "theta_exact")):
        claim(s, pc(c5["se_" + key]))
    for s, key in (("4,9", "iid"), ("22,9", "theta_hat"), ("25,6", "theta_exact")):
        claim(s, pc(c8["abdeckung_" + key]))
    for s, key in (("0,5", "iid"), ("0,9", "theta_hat"), ("1,0", "theta_exact")):
        claim(s, pc(c8["se_" + key]))
    assert c5["k_median"] == 150 and c8["k_median"] == 150 and c5["R"] == c8["R"] == 2000
    claim("37,5", c5["k_median"] * c5["theta_exact"])
    claim("0,297", c5["theta_hat_median"])
    claim("0,063", c8["theta_hat_median"])
    claim("−0,323", c8["verzerrung_ln"])
    claim("−0,085", c5["verzerrung_ln"])
    claim("90", pc(c8["anteil_unter_wahr"]), tol_extra=0.0)
    claim("73", pc(c5["anteil_unter_wahr"]))
    assert c8["breite_theta_exact"] / c8["breite_iid"] == pytest.approx(5.0, rel=1e-6) and c5["breite_theta_exact"] / c5["breite_iid"] == pytest.approx(2.0, rel=1e-6)
    assert "×2,0 (Last 50 %) und ×5,0 (Last 80 %)" in README and "Konstruktion: die Varianz wird durch θ geteilt, kein Messbefund" in README


def test_streuung_gegen_angenommene_standardabweichung():
    """Die Streuung von ln des Pegels (Standardabweichung der Läufe) gegen die Standardabweichung, die das Intervall mit exaktem θ unterstellt (Median der halben Breite durch 1,96)."""
    z = 1.959964
    c5, c8 = cell(0.5), cell(0.8)
    claim("0,194", c5["sd_ln"])
    claim("0,240", c8["sd_ln"])
    claim("0,099", c5["breite_theta_exact"] / (2 * z))
    claim("0,087", c8["breite_theta_exact"] / (2 * z))
    claim("2,0", c5["sd_ln"] / (c5["breite_theta_exact"] / (2 * z)))
    claim("2,7", c8["sd_ln"] / (c8["breite_theta_exact"] / (2 * z)))
    cells = REF.coverage()["abdeckung"]
    assert min(c["sd_ln"] / (c["breite_theta_exact"] / (2 * z)) for c in cells) >= 1.6 and "in allen 36 Zellen mindestens das 1,6-Fache" in README
    assert "2,0- und 2,7-Fache" in README and "3,3-Fache" in README
    for s in ("rund 89 %", "rund 93 %", "(Test, Band 2,3 bis 4,2)"):
        assert s in README                                                                      # Zahlen stammen aus `test_oracle_gpd.py` (Simulation, Band)


def test_abdeckung_korrekturtabelle_gegen_vormessung():
    c5, c8 = cell(0.5), cell(0.8)
    claim("1,1", pc(c5["se_iid"]))
    for new, old in ((pc(c5["abdeckung_iid"]), 70.4), (pc(c5["abdeckung_theta_hat"]), 93.2), (pc(c5["abdeckung_theta_exact"]), 96.4),
                     (pc(c8["abdeckung_iid"]), 16.8), (pc(c8["abdeckung_theta_hat"]), 72.0), (pc(c8["abdeckung_theta_exact"]), 88.2)):
        assert old - new > 11.0                                                                         # die Vormessung (Vorzeichenfehler) liegt weit über der korrigierten Messung
    assert "70,4 / 93,2 / 96,4 %" in README and "16,8 / 72,0 / 88,2 %" in README
    claim("0,297", c5["theta_hat_median"])
    assert "0,295 / 0,064" in README


def test_alle_36_zellen():
    cells = REF.coverage()["abdeckung"]
    iid = [pc(c["abdeckung_iid"]) for c in cells]
    hat = [pc(c["abdeckung_theta_hat"]) for c in cells]
    ex = [pc(c["abdeckung_theta_exact"]) for c in cells]
    claim("1,0", min(iid))
    claim("45,4", max(iid))
    claim("3,3", min(hat))
    claim("69,9", max(hat))
    claim("10,2", min(ex), tol_extra=1e-6)
    claim("74,2", max(ex), tol_extra=1e-6)
    assert "1,0 bis 45,4 %, mit θ̂ 3,3 bis 69,9 %, mit exaktem θ 10,2 bis 74,2 %: in keiner Zelle erreicht eine der drei Varianten 93 %" in README and max(iid + hat + ex) < 93
    big = [c for c in cells if c["rho"] == 0.5 and c["n"] == 12000 and c["anteil"] == 0.1 and c["p"] == 1e-3][0]
    assert big["k_median"] * big["theta_exact"] == pytest.approx(300.0)
    claim("71,7", pc(big["abdeckung_theta_exact"]))
    assert all(c["theta_hat_median"] > c["theta_exact"] for c in cells) and "in 36 von 36 Zellen" in README
    assert all(c["abdeckung_theta_hat"] < c["abdeckung_theta_exact"] for c in cells) and "mit θ̂ deckt das Intervall in 36 von 36 Zellen weniger als mit exaktem θ" in README
    pairs = [(a, b) for a in cells for b in cells if (a["rho"], a["n"], a["anteil"]) == (b["rho"], b["n"], b["anteil"]) and a["p"] == 1e-4 and b["p"] == 1e-3]
    assert len(pairs) == 18 and sum(a["abdeckung_theta_exact"] < b["abdeckung_theta_exact"] for a, b in pairs) == 13 and "in 13 von 18 Paaren" in README
    for printed in ("4,2 bis 76,9 % / 15,4 bis 97,0 % / 37,3 bis 98,5 %", "13 von 36 Zellen mindestens 93 %"):
        assert printed in README                                                                       # Zahlen der ersten Fassung (falsches Vorzeichen), eingefroren in `test_erste_fassung_...`
    claim("62", min(pc(c["anteil_unter_wahr"]) for c in cells), tol_extra=0.5)
    claim("99", max(pc(c["anteil_unter_wahr"]) for c in cells), tol_extra=0.5)
    assert all(c["verzerrung_ln"] < 0 for c in cells) and "in allen 36 Zellen" in README


def test_negativbefund_last_80_und_150_ueberschreitungen():
    c8 = cell(0.8)
    claim("25,6", pc(c8["abdeckung_theta_exact"]))
    claim("61,7", pc(cell(0.5)["abdeckung_theta_exact"]))
    assert "(6 Cluster) deckt sie 25,6 %" in README and c8["k_median"] * c8["theta_exact"] == pytest.approx(6.0)


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------------
def _est(model, a, q):
    for r in REF.estimator_rows():
        if model == "maxar" and r["modell"] == "maxar" and r["phi"] == a and r["quantil"] == q:
            return r
        if model == "mm1" and r["modell"] == "mm1" and r["rho"] == a and r["randanteil"] == q:
            return r
    raise KeyError((model, a, q))


def test_schaetzer_max_ar1():
    r0 = _est("maxar", 0.0, 0.95)
    claim("0,597", r0["runs_r10"]["median"])
    claim("1,00", r0["bloecke_b50"]["median"])
    claim("1,00", r0["intervalle"]["median"])
    assert r0["R"] == 500 and r0["n"] == 5000
    for phi, s in ((0.5, ("0,383", "0,510", "0,499")), (0.8, ("0,179", "0,220", "0,206"))):
        r = _est("maxar", phi, 0.95)
        for key, printed in zip(("runs_r10", "bloecke_b50", "intervalle"), s):
            claim(printed, r[key]["median"])
    claim("0,912", _est("maxar", 0.0, 0.99)["runs_r10"]["median"])


def test_schaetzer_mm1():
    for rho, vals, kern in ((0.5, ("0,26", "0,29", "0,27"), "0,26"), (0.8, ("0,05", "0,07", "0,04"), "0,04"), (0.9, ("0,02", "0,04", "0,01"), "0,01")):
        r = _est("mm1", rho, 0.01)
        for key, printed in zip(("runs_r10", "bloecke_b50", "intervalle"), vals):
            claim(printed, r[key]["median"])
        claim(kern, r["theta_kern_h02"])
        assert r["R"] == 40 and r["n"] == 100000
    r = _est("mm1", 0.9, 0.001)
    assert r["k_median"] == 22.5 and r["runs_r10"]["n"] == 28 and "12 von 40 Läufen" in README and "hat erwartet 100 Überschreitungen" in README


# --- Block gegen Schwelle ---------------------------------------------------------------------------------------------------------------------
def test_block_gegen_schwelle():
    rows = {r["phi"]: r for r in REF.coverage()["block_schwelle"]}
    for phi, s in zip((0.0, 0.5, 0.8), ("88,9", "86,1", "84,3")):
        claim(s, pc(rows[phi]["block_gev_pegelskala"]["abdeckung"]))
    for phi, s in zip((0.0, 0.5, 0.8), ("93,1", "93,8", "91,6")):
        claim(s, pc(rows[phi]["block_gev_lnskala"]["abdeckung"]))
    for phi, s in zip((0.0, 0.5, 0.8), ("89,0", "62,4", "32,6")):
        claim(s, pc(rows[phi]["k50"]["bereinigt_theta_1"]["abdeckung"]))
    for phi, s in zip((0.0, 0.5, 0.8), ("89,0", "75,6", "60,5")):
        claim(s, pc(rows[phi]["k50"]["bereinigt_theta_exakt"]["abdeckung"]))
    for phi, s in zip((0.0, 0.5, 0.8), ("92,2", "68,5", "38,7")):
        claim(s, pc(rows[phi]["k125"]["bereinigt_theta_1"]["abdeckung"]))
    for phi, s in zip((0.0, 0.5, 0.8), ("92,2", "81,9", "72,1")):
        claim(s, pc(rows[phi]["k125"]["bereinigt_theta_exakt"]["abdeckung"]))
    for s in ("95,3 / 78,3 / 49,1 %", "95,3 / 89,5 / 82,0 %", "95,9 / 92,5 / 87,4 %", "95,8 / 80,3 / 50,2 %"):
        assert s in README                                                                             # erste Fassung und Vormessung, nur als Vergleich zitiert
    assert pc(rows[0.8]["block_gev_pegelskala"]["abdeckung"]) > pc(rows[0.8]["k50"]["bereinigt_theta_exakt"]["abdeckung"]) + 20       # Block weit vor Schwelle bei φ = 0,8
    assert pc(rows[0.5]["block_gev_pegelskala"]["abdeckung"]) > pc(rows[0.5]["k50"]["bereinigt_theta_exakt"]["abdeckung"]) + 8
    assert [rows[p]["k50"]["cluster"] for p in (0.0, 0.5, 0.8)] == pytest.approx([50, 25, 10]) and [rows[p]["k125"]["cluster"] for p in (0.0, 0.5, 0.8)] == pytest.approx([125, 62.5, 25])
    assert "50 / 25 / 10 (125: 125 / 62,5 / 25)" in README and all(r["R"] == 1500 for r in rows.values())
    claim("83,3", pc(rows[0.8]["k50"]["alt_theta_exakt"]["abdeckung"]))
    claim("87,7", pc(rows[0.5]["k50"]["alt_theta_exakt"]["abdeckung"]))
    assert "deckt 83,3 und 87,7 % bei φ = 0,8 und 0,5" in README
    assert abs(rows[0.8]["k50"]["verhaeltnis_alt_median"] - 1.0) < 0.05
    assert "0,55 / 0,42 des Ziels bei φ = 0,5 / 0,8 in Experiment 5, 0,82 bei φ = 0" in README
    for phi, v in zip((0.0, 0.5, 0.8), (0.82, 0.55, 0.42)):
        assert abs(rows[phi]["k50"]["verhaeltnis_bereinigt_median"] - v) <= 0.005 + 1e-9


def test_block_gegen_schwelle_korrekturtabelle_im_rahmen():
    rows = {r["phi"]: r for r in REF.coverage()["block_schwelle"]}
    assert 80.3 - pc(rows[0.5]["k50"]["bereinigt_theta_1"]["abdeckung"]) > 10 and 90.8 - pc(rows[0.5]["k50"]["bereinigt_theta_exakt"]["abdeckung"]) > 10      # Vormessung mit Vorzeichenfehler
    assert abs(pc(rows[0.0]["block_gev_lnskala"]["abdeckung"]) - 95.3) / pc(rows[0.0]["block_gev_lnskala"]["se"]) > 3.0                  # ehrlich gemeldete Abweichung der ln-Skala bei φ = 0


def test_ziel_und_niveau():
    rows = {r["phi"]: r for r in REF.coverage()["block_schwelle"]}
    p_obs = 1 - 0.99 ** (1 / 50)
    for phi, s in zip((0.0, 0.5, 0.8), ("2,01", "3,94", "9,30")):
        claim(s, rows[phi]["p_stern"] * 1e4)
    claim("2,01", p_obs * 1e4)
    claim("1,96", rows[0.5]["p_stern"] / p_obs)
    claim("4,63", rows[0.8]["p_stern"] / p_obs)
    assert "x\\* = s_b/(−ln 0,99)" in README


# --- Quellen, Aussagen über die Herleitung ----------------------------------------------------------------------------------------------------
def test_quelle_theta_bleibt_stufe_a_und_die_herleitung_ist_eigene_arbeit():
    for phrase in ("der Wortlaut ist nicht gelesen", "eigene Arbeit", "Stufe A", "Hooghiemstra, Meester (1996)", "nicht gelesen, nicht benutzt", "Keine Neuheit"):
        assert phrase in README
    assert "Peshkova" in README and README.index("Peshkova") < README.index("eigene Arbeit")


def test_readme_nennt_die_pflichtabschnitte_und_hat_keine_platzhalter():
    for h in ("## Warum dieses Problem", "## Befunde und Korrekturen", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## 🚧 Bewusst nicht enthalten / nicht gemessen",
              "## Quellen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht umgesetzt", "## Lokal ausführen", "## Tests ausführen"):
        assert h in README
    assert "PLATZHALTER" not in README and "TODO: " not in README.replace("<!-- TODO nach dem Deploy", "")


def test_regler_und_dateien_der_readme_gibt_es():
    root = Path(__file__).resolve().parent.parent
    for name in re.findall(r"`((?:tools/)?clu_[a-z_]+\.(?:py|json))`", README):
        assert (root / name).exists() or (root / "tools" / name).exists(), name
    assert (root / "app.py").exists() and C.PRESETS
    assert float(np.max(np.array(C.RHO1_MAX))) < 1
