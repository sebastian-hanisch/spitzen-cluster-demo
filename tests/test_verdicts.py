"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import numpy as np

import clu_evaluation as E


def _series(k, clusters_expected=40.0, n_clusters=34, rho=0.5):
    return E.SeriesAnalysis(rho=rho, n=3000, f_bar=0.05, r=10, b=50, x=np.zeros(3), u=6.0, k=k, clusters=tuple(np.array([i]) for i in range(n_clusters)), theta_runs=0.212, theta_blocks=0.278,
                            theta_intervals=0.257, theta_exact=0.25, theta_kernel=0.2917, mean_cluster_exact=4.0, clusters_expected=clusters_expected)


def test_verdict_series_nennt_theta_clustergroesse_und_zaehlung():
    kind, text = E.verdict_series(_series(160))
    assert kind == "info"
    assert "Last 50 %" in text and "θ = 0,2500" in text and "im Mittel 4 Überschreitungen je Cluster" in text and "θ(u) = 0,2917" in text
    assert "160 Überschreitungen" in text and "k·θ = 40,0 unabhängige Cluster" in text and "zählt hier 34 Cluster" in text
    assert "Runs 0,212" in text and "Blöcke (b = 50) 0,278" in text and "Intervalle 0,257" in text


def test_verdict_series_schwelle_der_warnung_bei_zehn_clustern():
    assert E.verdict_series(_series(160, clusters_expected=10.0))[0] == "info"            # genau 10: noch keine Warnung
    kind, text = E.verdict_series(_series(160, clusters_expected=9.9))
    assert kind == "warning" and "k·θ = 9,9" in text


def test_verdict_series_ohne_ueberschreitung_ist_eine_warnung():
    kind, text = E.verdict_series(_series(0, clusters_expected=0.0, n_clusters=0))
    assert kind == "warning" and "kein Kunde über der Schwelle" in text and "3.000 Kunden" in text and "5 %" in text


def _rows():
    def r(phi, q, runs, blocks, ints):
        return {"modell": "maxar", "phi": phi, "quantil": q, "n": 5000, "R": 500, "theta_exact": 1 - phi,
                "runs_r10": {"median": runs}, "bloecke_b50": {"median": blocks}, "intervalle": {"median": ints}}
    return [r(0.0, 0.95, 0.597, 1.0, 1.0), r(0.8, 0.95, 0.179, 0.22, 0.206)]


def test_verdict_estimators_nennt_die_unabhaengigen_daten_und_phi_08():
    kind, text = E.verdict_estimators(_rows())
    assert kind == "info"
    assert "φ = 0, 5.000 Beobachtungen, 5 % über der Schwelle, 500 Läufe" in text and "Median 0,60 statt 1,0" in text and "(b = 50) 1,00, die Intervalle 1,00" in text
    assert "Mediane 0,18 / 0,22 / 0,21" in text
    assert "sobald ihr Abstand größer als 10 Kunden ist" in text and "mehr als 10 Kunden dazwischen" not in text


def _day(q_exact, q_theta, q_iid=20.0, n=256, rho=0.8, q=0.5):
    return E.DayView(rho=rho, n=n, q=q, xs=np.zeros(2), exact=np.zeros(2), gumbel_theta=np.zeros(2), gumbel_iid=np.zeros(2), q_exact=q_exact, q_theta=q_theta, q_iid=q_iid,
                     shift=q_iid - q_exact, shift_limit=16.09, independent_clusters=8.2)


def test_verdict_day_zahlen_und_schwelle_zehn_prozent():
    kind, text = E.verdict_day(_day(14.0, 12.3, 28.4))
    assert kind == "warning"                                                          # 12,1 % daneben
    assert "Last 80 %, 256 Kunden, 0,5-Quantil" in text and "exakt 14,0" in text and "unabhängig gerechnet 28,4" in text and "Verschiebung 14,4 Abfertigungsdauern" in text
    assert "≈ 43 min" in text and "Der Grenzwert ln(1/θ)/γ ist 16,1 (≈ 48 min)" in text and "−12 % gegen exakt" in text and "n·θ·ρ = 8,2" in text
    assert E.verdict_day(_day(10.0, 11.0, 28.4))[0] == "info"                         # genau +10 %: noch brauchbar
    assert E.verdict_day(_day(10.0, 11.01, 28.4))[0] == "warning"
    assert "+10 % gegen exakt" in E.verdict_day(_day(10.0, 11.01, 28.4))[1]


def _cell(iid, hat, ex, k=150.0, R=2000):
    return {"abdeckung_iid": iid, "abdeckung_theta_hat": hat, "abdeckung_theta_exact": ex, "se_iid": 0.009, "k_median": k, "theta_exact": 0.04, "R": R, "sd_ln": 0.24, "breite_theta_exact": 0.3427}


def test_verdict_coverage_zahlen_und_arten():
    kind, text = E.verdict_coverage(_cell(0.179, 0.702, 0.869), 0.8, 3000, 0.05, 1e-3)
    assert kind == "warning"
    assert "Last 80 %, 3.000 Kunden, Schwelle bei 5 % der Kunden (im Median 150 Überschreitungen)" in text and "p = 10⁻³ je Kunde, 2.000 Läufe" in text
    assert "in 17,9 % (Standardfehler 0,9 Punkte)" in text and "mit geschätztem θ in 70,2 %" in text and "exakten θ = 0,040 in 86,9 %" in text and "k·θ = 6,0 unabhängige Cluster" in text
    assert "nirgends 93 %" in text and "Standardabweichung von ln des Pegels 0,240 gegen 0,087" in text and "nur, wenn genug Cluster da sind" not in text
    assert E.verdict_coverage(_cell(0.95, 0.95, 0.95), 0.5, 1000, 0.1, 1e-3)[0] == "success"
    assert E.verdict_coverage(_cell(0.70, 0.93, 0.95), 0.5, 1000, 0.1, 1e-3)[0] == "info"
    assert E.verdict_coverage(_cell(0.70, 0.90, 0.9299), 0.5, 1000, 0.1, 1e-3)[0] == "warning"       # Schwelle 93 %: knapp darunter
    assert E.verdict_coverage(_cell(0.70, 0.90, 0.93), 0.5, 1000, 0.1, 1e-3)[0] == "info"


def test_verdict_blocks_zahlen_ohne_neu_zu_rechnen():
    kind, text = E.verdict_blocks(0.8, 256)
    assert kind == "info"
    assert "bei 256 Kunden je Block hat die beste GEV (kleinster Kolmogorov-Abstand) ξ = +0,087" in text and "64 / 256 / 1.024 / 4.096 Kunden je Block: ξ = +0,159 / +0,087 / +0,043 / +0,018" in text
    assert "σ = 4,12 liegt unter dem Grenzwert 1/γ = 5,0" in text and "Kolmogorov-Abstand 0,0120 gegen 0,0029" in text
    assert "Populationswert der Likelihood-Anpassung liegt bei dieser Blocklänge über null" in text


def _hrow():
    cov = lambda m, se=0.01: {"abdeckung": m, "se": se, "n": 1500}
    return {"phi": 0.8, "theta": 0.2, "R": 1500, "block_gev_pegelskala": cov(0.843), "block_gev_lnskala": cov(0.916),
            "k50": {"cluster": 10.0, "bereinigt_theta_1": cov(0.491), "bereinigt_theta_exakt": cov(0.82), "alt_theta_exakt": cov(0.955)},
            "k125": {"cluster": 25.0, "bereinigt_theta_1": cov(0.535), "bereinigt_theta_exakt": cov(0.934), "alt_theta_exakt": cov(0.967)}}


def test_verdict_block_schwelle_zahlen_und_schwelle():
    kind, text = E.verdict_block_schwelle(_hrow(), 50)
    assert kind == "warning"
    assert "φ = 0,8 (θ = 0,2)" in text and "1.500 Läufe" in text and "Block-GEV deckt in 84,3 %" in text and "ln-Skala 91,6 %" in text
    assert "50 Überschreitungen (rechnerisch 10,0 Cluster) deckt ohne θ in 49,1 %, mit dem exakten θ in 82,0 %" in text
    kind125, text125 = E.verdict_block_schwelle(_hrow(), 125)
    assert kind125 == "success" and "125 Überschreitungen (rechnerisch 25,0 Cluster)" in text125 and "93,4 %" in text125
    assert "weniger als der POT-Ansatz" in text and "bleibt die Abhängigkeit ohne Folge" not in text
