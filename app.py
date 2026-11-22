"""Spitzen-Cluster: der Extremalindex der Wartezeit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünf Experimente: (1) eine Reihe, ihre Spitzen, ihre Cluster, (2) das Tagesmaximum exakt gegen „unabhängig gerechnet“, (3) hält das 95-%-Intervall, (4) wie groß muss der Block sein,
(5) Block gegen Schwelle bei Abhängigkeit. Zufall nur aus der SplitMix64-Vorlage; die schweren Rechnungen sind eingefroren (`clu_*.json`, Erzeuger `tools/clu_make_*.py`).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import clu_constants as C
import clu_evaluation as E
import clu_kernel as K
import clu_reference as REF
from clu_layout import dunkles_design, kopf_css
from clu_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from clu_texts import de, frac_text, p_text, pct, q_text, rho_text, thousands
from clu_visualization import build_block_schwelle, build_blocks, build_blocks_xi, build_coverage, build_day_cdf, build_series

st.set_page_config(page_title="Spitzen-Cluster – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _series(rho, n, f_bar, r, b, seed):
    return E.analyse_series(rho, n, f_bar, r, b, seed)


@st.cache_data(show_spinner=False)
def _live(rho, n, frac, p, R, seed):
    return E.live_coverage(rho, n, frac, p, R, seed)


def show(verdict):
    """Zeigt ein Verdikt (Art, Text) mit dem passenden Streamlit-Element."""
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in clu_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Spitzen-Cluster: der Extremalindex der Wartezeit")
st.markdown(
    """
Hohe Wartezeiten kommen **im Gleichschritt**: Wer in einer Warteschlange eine sehr lange Wartezeit sieht, sieht gleich danach mehrere. Der **Extremalindex θ** (nicht der Extremwertindex ξ) misst das:
**1/θ** ist im Grenzwert hoher Schwellen die mittlere Zahl von Überschreitungen je Cluster (je Spitze); ein Cluster kann Lücken enthalten und mehr Kunden überspannen. Im Modell der Seite, einer **M/M/1-Schlange** mit unendlichem Warteraum, ist **θ = (1 − Last)²**,
also 1/4, 1/25 und 1/100 bei 50, 80 und 90 % Last. Wer die Kunden eines Tages wie **unabhängige** Beobachtungen behandelt, verschiebt das Tagesmaximum und lässt Intervalle für den Pegel ihr Versprechen
brechen: **3.000 Kunden sind keine Stichprobe vom Umfang 3.000.** Fünf Experimente zeigen das mit exakten Rechnungen und Läufen. Das Modell ist **ausgedacht**: gezeigt wird, was in ihm geschieht,
nicht, was das für einen echten Betrieb heißt. **Stationär, aber abhängig:** die Verteilung ändert sich nicht, die Werte hängen aneinander.

Wer das Phänomen zuerst ohne Fachbegriffe verstehen will, öffnet **💬 In einfachen Worten**. Weiter unten: **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/spitzen-cluster-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Hohe Wartezeiten kommen im Gleichschritt: Nach einer sehr langen Wartezeit an einer Kasse folgen meist gleich weitere, bis die Schlange abgebaut ist. Rechnet man trotzdem so, als wären alle Kunden unabhängige Beobachtungen, schätzt man den schlimmsten Wert eines Tages zu hoch und ein Sicherheitsband für den seltenen Wert zu schmal. Die Demo zählt die Pulks, die der Extremalindex misst, rechnet das Tagesmaximum ohne Zufall aus einem feinen Gitter und prüft in Läufen, wie oft das Band sein Versprechen hält. Der Haken: Auch mit dem exakten Wert für die Pulks hält das Band sein Versprechen in keinem der durchgespielten Fälle. Alles gilt für ein ausgedachtes Modell, nicht für einen echten Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/spitzen-cluster-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Seite", expanded=True):
    st.markdown(
        """
1. **Das Modell.** Dienstrate 1 (die Abfertigungsdauer ist die Zeiteinheit), Ankunftsrate ρ (die Last); die Wartezeit des nächsten Kunden ist W' = max(0, W + Bedienzeit − Zwischenankunftszeit), beide exponentiell.
2. **Spitzen und Cluster.** Eine Schwelle u macht aus der Reihe Überschreitungen; liegen zwei Überschreitungen höchstens r Kunden auseinander, gehören sie zum selben Cluster (Runs-Regel). Gezählt wird in Clustern, nicht in Überschreitungen.
3. **Die Wahrheit.** Ein exakter Rechenkern (Gitter) liefert die Verteilung des Tagesmaximums und θ(u); die geschlossene Formel θ = (1 − ρ)² ist der Grenzwert hoher Schwellen.
4. **Was ein Schätzer sieht.** Die Schätzer von θ (Runs, Blöcke, Intervalle) und das Intervall für den Pegel arbeiten an Läufen; ihre Abweichung von der Wahrheit ist der Befund.
        """
    )

st.markdown("## ❓ Kernfragen")
st.markdown(
    """
1. **Wie stark häufen sich die Spitzen?** (Experiment 1)
2. **Was kostet es, sie wie unabhängig zu behandeln?** (Experiment 2)
3. **Wie oft hält ein 95-%-Intervall sein Versprechen?** (Experiment 3)
4. **Wie weit reicht der Gumbel-Grenzwert bei Tagesgröße, und welches ξ sieht eine Anpassung an Tagesmaxima?** (Experiment 4)
5. **Block oder Schwelle, wenn die Werte voneinander abhängen?** (Experiment 5)
"""
)
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Experiment 1: Eine Reihe**")
    rho1 = st.slider("Last ρ", C.RHO1_MIN, C.RHO1_MAX, step=C.RHO1_STEP, key="rho1", help="Ankunftsrate bei Dienstrate 1. Bis 0,9: der Kern ist nur bis dort geprüft; ab ρ = 1 ist die Schlange nicht stationär.")
    n1 = st.slider("Kunden n", *bounds("n1"), step=C.N1_STEP, key="n1", help="Zahl aufeinanderfolgender Kunden der Reihe (stationär gestartet).")
    frac1 = st.select_slider("Anteil der Kunden über der Schwelle", options=C.FRAC1_OPTIONS, key="frac1", format_func=frac_text,
                             help="Die Schwelle u liegt dort, wo die Wartezeit mit dieser Wahrscheinlichkeit überschritten wird (exakt aus der Randverteilung).")
    r1 = st.slider("Runs-Regel r (Kunden)", *bounds("r1"), key="r1", help="Zwei Überschreitungen mit höchstens r Kunden Abstand gehören zum selben Cluster.")
    b1 = st.slider("Blocklänge b (Kunden)", *bounds("b1"), step=C.B1_STEP, key="b1", help="Blocklänge des Blöcke-Schätzers von θ.")
    st.markdown("**Experiment 2: Das Tagesmaximum**")
    rho2 = st.select_slider("Last ρ ", options=C.RHO2_OPTIONS, key="rho2", format_func=rho_text, help="Eingefrorene exakte Verteilungen für drei Lasten.")
    n2 = st.select_slider("Kunden je Tag n", options=C.N2_OPTIONS, key="n2", format_func=thousands, help="Das Tagesmaximum ist die größte Wartezeit unter n aufeinanderfolgenden Kunden.")
    q2 = st.select_slider("Quantil", options=C.Q2_OPTIONS, key="q2", format_func=q_text, help="Welches Quantil des Tagesmaximums verglichen wird.")
    st.markdown("**Experiment 3: Hält das Intervall?**")
    rho3 = st.select_slider("Last ρ  ", options=C.RHO3_OPTIONS, key="rho3", format_func=rho_text, help="Eingefrorene Studie: zwei Lasten.")
    n3 = st.select_slider("Kunden n je Lauf", options=C.N3_OPTIONS, key="n3", format_func=thousands, help="Aufeinanderfolgende Kunden je Lauf.")
    frac3 = st.select_slider("Anteil über der Schwelle (Läufe)", options=C.FRAC3_OPTIONS, key="frac3", format_func=frac_text, help="Die Schwelle ist das empirische Quantil der Reihe; k = Anteil · n Überschreitungen.")
    p3 = st.select_slider("Pegel zu p je Kunde", options=C.P3_OPTIONS, key="p3", format_func=p_text, help="Gesucht ist der Wert, den ein Kunde mit Wahrscheinlichkeit p überschreitet (nicht die Wiederkehrperiode eines Tages).")
    r3 = st.select_slider("Live-Läufe R (0 = aus)", options=C.R3_OPTIONS, key="r3", help="Läufe, die in dieser Sitzung mit dem Zufalls-Seed unten gerechnet werden; die eingefrorene Studie hat 2.000 je Zelle.")
    st.markdown("**Experiment 4: Blocklänge**")
    rho4 = st.select_slider("Last ρ   ", options=C.RHO4_OPTIONS, key="rho4", format_func=rho_text, help="Eingefrorene exakte Verteilungen des Blockmaximums.")
    n4 = st.select_slider("Kunden je Block", options=C.N4_OPTIONS, key="n4", format_func=thousands, help="Blocklänge in Kunden.")
    st.markdown("**Experiment 5: Block gegen Schwelle**")
    phi5 = st.select_slider("φ des max-AR(1) (θ = 1 − φ)", options=C.PHI5_OPTIONS, key="phi5", format_func=lambda v: de(v, 1), help="φ = 0 sind unabhängige Werte; je größer φ, desto kleiner θ.")
    k5 = st.select_slider("Überschreitungen k", options=C.K5_OPTIONS, key="k5", help="Zahl der größten Werte, an die der POT-Ansatz (Peaks over Threshold) die GPD anpasst.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für die Reihe aus Experiment 1 und die Live-Läufe aus Experiment 3.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"rho1": float(rho1), "n1": int(n1), "frac1": float(frac1), "r1": int(r1), "b1": int(b1), "rho2": float(rho2), "n2": int(n2), "q2": float(q2), "rho3": float(rho3),
                   "n3": int(n3), "frac3": float(frac3), "p3": float(p3), "r3": int(r3), "rho4": float(rho4), "n4": int(n4), "phi5": float(phi5), "k5": int(k5), "seed_input": int(seed)})

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Eine Reihe, ihre Spitzen, ihre Cluster")
st.caption("Eine Reihe der M/M/1-Wartezeit (Zufalls-Seed in der Seitenleiste); die Schwelle hat den exakten Randanteil aus der Seitenleiste. Eine einzelne Reihe ist kein Befund: sie zeigt, wie die Rechnung aussieht.")
a = _series(float(rho1), int(n1), float(frac1), int(r1), int(b1), int(seed))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Überschreitungen k", thousands(a.k), delta=f"Schwelle u = {de(a.u, 1)}", delta_color="off")
m2.metric("Cluster (Runs, r = %d)" % a.r, thousands(a.n_clusters), delta=(f"{de(a.k / a.n_clusters, 1)} Überschreitungen je Cluster" if a.n_clusters else "keine Überschreitung"), delta_color="off")
m3.metric("Überschreitungen je Cluster (1/θ, Grenzwert)", de(a.mean_cluster_exact, 0), delta=f"θ = {de(a.theta_exact, 4)}", delta_color="off")
m4.metric("Unabhängige Cluster k·θ", de(a.clusters_expected, 1), delta=f"unter k = {thousands(a.k)} Überschreitungen", delta_color="off")
t1, t2, t3, t4 = st.columns(4)
t1.metric("θ(u) aus dem Kern", de(a.theta_kernel, 4), delta="Gitter h = 0,2, zu dieser Schwelle", delta_color="off")
t2.metric("θ̂ Runs (r = %d)" % a.r, de(a.theta_runs, 3), delta=f"exakt {de(a.theta_exact, 4)}", delta_color="off")
t3.metric("θ̂ Blöcke (b = %d)" % a.b, de(a.theta_blocks, 3), delta=f"exakt {de(a.theta_exact, 4)}", delta_color="off")
t4.metric("θ̂ Intervalle", de(a.theta_intervals, 3), delta=f"exakt {de(a.theta_exact, 4)}", delta_color="off")
st.plotly_chart(build_series(a), width="stretch", key="series_chart")
st.caption("Farbig: die Überschreitungen, gleiche Farbe = gleicher Cluster (die Farben wiederholen sich nach acht Clustern). Gestrichelt: die Schwelle.")
show(E.verdict_series(a))

st.markdown("**Was die Schätzer im Median sehen** (eingefrorene Läufe, Zähleinheit Läufe; max-AR(1) hat das exakte θ = 1 − φ, φ = 0 sind unabhängige Daten)")
maxar_rows = E.estimator_rows("maxar")
tab = pd.DataFrame(
    [{"Modell": f"max-AR(1), φ = {de(r['phi'], 1)}", "Schwelle: Anteil über u": frac_text(1 - r["quantil"]), "θ exakt": de(r["theta_exact"], 2), "Runs r = 10": de(r["runs_r10"]["median"], 3),
      "Blöcke b = 50": de(r["bloecke_b50"]["median"], 3), "Intervalle": de(r["intervalle"]["median"], 3), "Überschreitungen k (Median)": thousands(r["k_median"]),
      "Cluster (Runs, Median)": thousands(r["cluster_runs_median"])} for r in maxar_rows]
    + [{"Modell": f"M/M/1, Last {rho_text(r['rho'])}", "Schwelle: Anteil über u": frac_text(r["randanteil"]), "θ exakt": de(r["theta_exact"], 2), "Runs r = 10": de(r["runs_r10"]["median"], 3),
        "Blöcke b = 50": de(r["bloecke_b50"]["median"], 3), "Intervalle": de(r["intervalle"]["median"], 3), "Überschreitungen k (Median)": thousands(r["k_median"]),
        "Cluster (Runs, Median)": thousands(r["cluster_runs_median"])} for r in E.estimator_rows("mm1")]
)
st.dataframe(tab, hide_index=True, width="stretch")
st.caption(f"max-AR(1): {thousands(maxar_rows[0]['n'])} Beobachtungen je Lauf, {maxar_rows[0]['R']} Läufe je Zeile. M/M/1: {thousands(E.estimator_rows('mm1')[0]['n'])} Kunden je Lauf, {E.estimator_rows('mm1')[0]['R']} Läufe je Zeile; bei Last 90 % und Anteil 0,1 % hat nur ein Teil der Läufe genug Überschreitungen für alle Schätzer.")
show(E.verdict_estimators(maxar_rows))
st.caption("⚠️ **Falle:** Die Schätzer hängen an r, b und der Schwelle, und sie zählen Cluster, nicht Überschreitungen: sechs Überschreitungen im Abstand von höchstens zwei Kunden sind ein Cluster.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Das Tagesmaximum, unabhängig gerechnet gegen exakt")
st.caption("Tagesmaximum = größte Wartezeit unter n aufeinanderfolgenden Kunden (eine Annahme dieser Seite), stationär gestartet. Die exakte Verteilung kommt aus dem Gitterkern (h = 0,1, eingefroren, kein Zufall).")
dv = E.day_view(float(rho2), int(n2), float(q2))
d1, d2, d3, d4 = st.columns(4)
d1.metric("Quantil exakt", de(dv.q_exact, 1), delta=f"Quantil {q_text(dv.q)}, Abfertigungsdauern", delta_color="off")
d2.metric("Gumbel mit θ = (1−ρ)²", de(dv.q_theta, 1), delta=f"{de(dv.q_theta - dv.q_exact, 1)} gegen exakt".replace("-", "−"), delta_color="off")
d3.metric("Gumbel „unabhängig“ (θ = 1)", de(dv.q_iid, 1), delta=f"Verschiebung {de(dv.shift, 1)}", delta_color="off")
d4.metric("Cluster je Tag n·θ·ρ", de(dv.independent_clusters, 1), delta="unabhängige Cluster im Tag", delta_color="off")
st.plotly_chart(build_day_cdf(dv), width="stretch", key="day_chart")
qrows = []
for q in C.Q2_OPTIONS:
    ex = REF.day_quantile(float(rho2), int(n2), q)
    qt, qi = K.gumbel_day_quantile(float(rho2), int(n2), q), K.gumbel_day_quantile(float(rho2), int(n2), q, theta=1.0)
    qrows.append({"Quantil": q_text(q), "exakt": de(ex, 1), "Gumbel θ = (1−ρ)²": de(qt, 1), "Gumbel θ = 1": de(qi, 1), "Verschiebung (θ = 1 minus exakt)": de(qi - ex, 1)})
st.dataframe(pd.DataFrame(qrows), hide_index=True, width="stretch")
show(E.verdict_day(dv))
st.caption("⚠️ **Falle:** Die θ-Näherung stimmt am oberen Rand und im Körper der Verteilung nicht: sie braucht viele unabhängige Cluster je Tag. Bei wenigen Clustern (kleine n, hohe Last) kann sie sogar negative Quantile liefern.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Hält das 95-%-Intervall sein Versprechen?")
st.caption("Pegel zu p je Kunde, GPD-Anpassung (die GPD ist hier exakt: die Überschreitungen sind bei jeder Schwelle exponentiell), Delta-Intervall für ln x_p. Der Abdeckungsfehler kommt aus der Abhängigkeit und, bei wenigen Überschreitungen, aus der Näherung des Delta-Intervalls; die Demo misst beides nicht getrennt (an unabhängigen Daten deckt das Intervall in der Hauptzelle rund 89 %).")
cell = REF.coverage_cell(float(rho3), int(n3), float(frac3), float(p3))
c1, c2, c3 = st.columns(3)
c1.metric("Abdeckung: „unabhängig“ gerechnet", pct(cell["abdeckung_iid"]), delta=f"± {de(100 * cell['se_iid'], 1)} Punkte (Standardfehler)", delta_color="off")
c2.metric("Abdeckung: mit geschätztem θ", pct(cell["abdeckung_theta_hat"]), delta=f"± {de(100 * cell['se_theta_hat'], 1)} Punkte", delta_color="off")
c3.metric("Abdeckung: mit exaktem θ", pct(cell["abdeckung_theta_exact"]), delta=f"± {de(100 * cell['se_theta_exact'], 1)} Punkte", delta_color="off")
live = None
if int(r3) > 0:
    with st.spinner(f"{int(r3)} Läufe mit je {thousands(n3)} Kunden werden gerechnet (Streamlit Cloud: einige Sekunden) ..."):
        live = _live(float(rho3), int(n3), float(frac3), float(p3), int(r3), int(seed))
st.plotly_chart(build_coverage(cell, live), width="stretch", key="coverage_chart")
crow = {"Größe": ["Überschreitungen k (Median)", "Cluster unter k, k·θ (rechnerisch)", "Cluster der Runs-Regel r = 10 (Median)", "θ̂ (Median) gegen exakt", "Breite des Intervalls für ln x_p (Median)",
                  "Verzerrung ln(Pegel̂/Pegel)", "Anteil der Läufe mit Pegel̂ unter dem wahren"],
        "Eingefroren": [thousands(cell["k_median"]), de(cell["k_median"] * cell["theta_exact"], 1), thousands(cell["cluster_runs_median"]), f"{de(cell['theta_hat_median'], 3)} gegen {de(cell['theta_exact'], 3)}",
                        f"{de(cell['breite_iid'], 2)} / {de(cell['breite_theta_hat'], 2)} / {de(cell['breite_theta_exact'], 2)}", f"{de(cell['verzerrung_ln'], 3)} (sd {de(cell['sd_ln'], 3)})", pct(cell["anteil_unter_wahr"], 0)]}
if live is not None:
    crow["Live"] = [thousands(live["k_median"]), de(live["k_median"] * live["theta_exact"], 1), thousands(live["cluster_runs_median"]), f"{de(live['theta_hat_median'], 3)} gegen {de(live['theta_exact'], 3)}",
                    f"{de(live['breite_iid'], 2)} / {de(live['breite_theta_hat'], 2)} / {de(live['breite_theta_exact'], 2)}", f"{de(live['verzerrung_ln'], 3)} (sd {de(live['sd_ln'], 3)})", pct(live["anteil_unter_wahr"], 0)]
st.dataframe(pd.DataFrame(crow), hide_index=True, width="stretch")
show(E.verdict_coverage(cell, float(rho3), int(n3), float(frac3), float(p3)))
if live is not None:
    st.caption(f"Live: {live['R']} Läufe mit Zufalls-Seed {int(seed)}: {pct(live['abdeckung_iid'])} / {pct(live['abdeckung_theta_hat'])} / {pct(live['abdeckung_theta_exact'])} "
               f"(Standardfehler {de(100 * live['se_iid'], 1)} / {de(100 * live['se_theta_hat'], 1)} / {de(100 * live['se_theta_exact'], 1)} Punkte). Mit wenigen Läufen streut die Abdeckung: der Standardfehler gilt für diese Zahl, nicht für die Zeile oben.")
st.caption("⚠️ **Falle:** Abhängigkeit verschweigt sich: das Intervall sieht von außen wie jedes andere aus. Auch die Korrektur mit dem richtigen θ erreicht in dieser Studie nirgends 93 %: die Streuung des Pegels ist größer, als das Teilen der Varianz durch θ unterstellt, und das geschätzte θ liegt im Median über dem exakten (das Intervall wird noch enger).")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wie groß muss der Block sein?")
st.caption("Exakte Verteilung des Blockmaximums (Gitterkern, eingefroren) und die beste GEV daran. Gezeigt wird der Populationswert: das, wogegen eine Anpassung an unendlich viele Blöcke läuft.")
brow = REF.block_row(float(rho4), int(n4))
bs = E.block_summary(float(rho4))
b1c, b2c, b3c = st.columns(3)
b1c.metric("ξ der besten GEV (Kolmogorov-Abstand)", de(brow["ks_gev"]["xi"], 3), delta="Grenzwert: ξ = 0", delta_color="off")
b2c.metric("ξ der besten GEV (Likelihood)", de(brow["ml_gev"]["xi"], 3), delta="Populationswert der Maximum-Likelihood-Anpassung", delta_color="off")
b3c.metric("σ der besten GEV", de(brow["ks_gev"]["sigma"], 2), delta=f"Grenzwert 1/γ = {de(brow['sigma_grenzwert'], 1)}", delta_color="off")
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(build_blocks(float(rho4), int(n4), brow), width="stretch", key="blocks_chart")
with col_b:
    st.plotly_chart(build_blocks_xi(bs, float(rho4)), width="stretch", key="blocks_xi_chart")
st.dataframe(pd.DataFrame([{"Kunden je Block": thousands(r["n"]), "ξ (Kolmogorov)": de(r["xi_ks"], 3), "ξ (Likelihood)": de(r["xi_ml"], 3), "σ": de(r["sigma_ks"], 2),
                            "Kolmogorov-Abstand GEV": de(r["ks_gev"], 4), "Kolmogorov-Abstand Gumbel": de(r["ks_gumbel"], 4)} for r in bs]), hide_index=True, width="stretch")
show(E.verdict_blocks(float(rho4), int(n4)))
st.caption("⚠️ **Falle:** Eine Maximum-Likelihood-Anpassung an Tagesmaxima liefert ein ξ > 0 aus einer Wartezeit ohne schweren Rand. Das ist kein Rauschen, sondern der Populationswert bei dieser Blocklänge (bei Last 50 % und 4.096 Kunden nur +0,003); eine Anpassung an eine Stichprobe von Tagesmaxima streut um diesen Wert.")

st.markdown("---")

# --- Experiment 5 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Block gegen Schwelle, wenn die Werte voneinander abhängen")
st.caption("max-AR(1)-Prozess mit Fréchet(1)-Rand (exakte Blockverteilung, θ = 1 − φ): 50 Blöcke zu 50 Beobachtungen, Ziel der 100-Block-Wert. Eingefrorene Läufe; die GPD wird bei dem θ-bereinigten Niveau gerechnet.")
hrow = REF.block_schwelle_row(float(phi5))
h1c, h2c, h3c, h4c = st.columns(4)
kk = hrow[f"k{int(k5)}"]
h1c.metric("Block-GEV (Pegelskala)", pct(hrow["block_gev_pegelskala"]["abdeckung"]), delta=f"± {de(100 * hrow['block_gev_pegelskala']['se'], 1)} Punkte", delta_color="off")
h2c.metric("Schwelle ohne θ", pct(kk["bereinigt_theta_1"]["abdeckung"]), delta=f"± {de(100 * kk['bereinigt_theta_1']['se'], 1)} Punkte", delta_color="off")
h3c.metric("Schwelle mit exaktem θ", pct(kk["bereinigt_theta_exakt"]["abdeckung"]), delta=f"± {de(100 * kk['bereinigt_theta_exakt']['se'], 1)} Punkte", delta_color="off")
h4c.metric("Cluster unter k", de(kk["cluster"], 1), delta=f"k = {int(k5)}, θ = {de(hrow['theta'], 1)}", delta_color="off")
st.plotly_chart(build_block_schwelle(hrow, int(k5)), width="stretch", key="h_chart")
hrows = []
for r in REF.coverage()["block_schwelle"]:
    for k in C.K5_OPTIONS:
        v = r[f"k{k}"]
        hrows.append({"φ (θ = 1−φ)": f"{de(r['phi'], 1)} ({de(r['theta'], 1)})", "k": k, "Cluster k·θ": de(v["cluster"], 1), "Block-GEV, Pegelskala": pct(r["block_gev_pegelskala"]["abdeckung"]),
                      "Block-GEV, ln-Skala": pct(r["block_gev_lnskala"]["abdeckung"]), "Schwelle ohne θ": pct(v["bereinigt_theta_1"]["abdeckung"]), "Schwelle mit exaktem θ": pct(v["bereinigt_theta_exakt"]["abdeckung"]),
                      "Schwelle, erste Fassung (Niveau p = 1 − 0,99^(1/b))": pct(v["alt_theta_exakt"]["abdeckung"])})
st.dataframe(pd.DataFrame(hrows), hide_index=True, width="stretch")
show(E.verdict_block_schwelle(hrow, int(k5)))
st.caption("⚠️ **Falle:** Das Niveau des Ziels. Der 100-Block-Wert hat bei θ < 1 eine größere Überschreitungswahrscheinlichkeit je Beobachtung als 1 − 0,99^(1/b); wer die GPD beim falschen Niveau rechnet, "
           "trifft das Ziel scheinbar (letzte Spalte der Tabelle), obwohl das Intervall eine andere Größe überdeckt.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme der Seite | Was nicht gezeigt ist | Wer setzt an |
|---|---|---|
| **Ausgedachtes Modell:** eine Schlange, unendlicher Warteraum, gleichbleibende Last, Exponentialzeiten | Eine reale Wartezeitreihe, an der Spitzen-Quantile geschätzt und später geprüft wurden, liegt nicht vor. Was das Ergebnis für einen Betrieb heißt, wird nicht behauptet. | – |
| **Unendlicher Warteraum** | Ein endlicher Puffer (Gate) kappt die Cluster; in der Vormessung war θ dort deutlich größer, hier nicht neu gemessen. | – |
| **Stationär** (Verteilung und Last ändern sich nicht) | Trend und wechselnde Last: nicht hier. | Demo „Trend“ der Linie |
| **ξ = 0** (exponentieller Rand der Wartezeit) | Abhängigkeit zusammen mit einem schweren Rand (ξ > 0) und eine subexponentielle Bedienzeit (θ = 0 nach Literatur der Stufe A): nicht gemessen. | – |
| **Nur Pegel je Kunde und Tagesmaximum bei Last bis 90 %** | Last über 90 %, Reihen unter 1.000 Kunden und Intervalle für den Pegel des Tagesmaximums: nicht gemessen. | – |
| **Hill-Schätzer, Schwellenwahl, Block gegen Schwelle bei anderen Blocklängen unter Abhängigkeit** | nicht gemessen; diese Seite zeigt nur die Wirkung auf den Pegel bei fester Schwelle und die Zahl der Cluster. | – |
| **Quelle für θ = (1 − ρ)²** | Eigene Herleitung mit zwei Rechenwegen; die Quelle (Peshkova 2026) liegt nur als Kurzseite vor, der Wortlaut ist nicht gelesen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/spitzen-cluster-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die Last $\rho$ einer M/M/1-Schlange (Dienstrate 1, Ankunftsrate $\rho$). Gesucht sind der Extremalindex $\theta$ der Wartezeiten, die Verteilung ihres Tagesmaximums und die Abdeckung von Intervallen für den Pegel $x_p$.

| Symbol | Bedeutung |
|---|---|
| $W_n$ | Wartezeit des $n$-ten Kunden, in Abfertigungsdauern |
| $\rho$, $\gamma = 1-\rho$ | Last (Ankunftsrate bei Dienstrate 1) und Lundberg-Wurzel |
| $S_n$, $A_n$ | Bedienzeit (exponentiell mit Rate 1) und Zwischenankunftszeit (exponentiell mit Rate $\rho$) |
| $\theta$ | Extremalindex: Kehrwert der mittleren Clustergröße im Grenzwert hoher Schwellen (nicht der Extremwertindex $\xi$) |
| $u$, $k$, $\zeta$ | Schwelle, Zahl der Überschreitungen und Anteil $\zeta = k/n$ über der Schwelle |
| $x_p$, $x_q$ | Pegel zur Überschreitungswahrscheinlichkeit $p$ je Kunde, Quantil $q$ des Tagesmaximums |
| $\pi$, $Q$ | stationäre Verteilung und Kern der Kette auf den Zuständen bis $x$ (Gitter) |
| $\sigma$, $\xi$ | Skala und Form der angepassten Verteilung (GPD) |
| $B$, $b$, $K$, $N$ | Blöcke: Zahl und Länge, Zahl der Blöcke mit Überschreitung, Zahl der Überschreitungen |
| $\varphi$, $M_b$, $s_b$ | Parameter des max-AR(1), Maximum über $b$ Werte und dessen Skala |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/spitzen-cluster-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben ist die Last $\rho \in (0,1)$ einer M/M/1-Schlange (Dienstrate 1, Ankunftsrate $\rho$); gesucht sind der Extremalindex $\theta$ der Wartezeiten, die Verteilung ihres Maximums und die Abdeckung von Intervallen für Pegel.
Die Formeln sind das vollständige Modell der Simulation und eine **Skizze** der Herleitung von $\theta$ (die Herleitung ist eigene Arbeit, siehe unten).

| Symbol | Bedeutung |
|---|---|
| $W_n$ | Wartezeit des $n$-ten Kunden, in Abfertigungsdauern |
| $\rho$, $\gamma = 1-\rho$ | Last (Ankunftsrate bei Dienstrate 1) und Lundberg-Wurzel |
| $S_n \sim \mathrm{Exp}(1)$, $A_n \sim \mathrm{Exp}(\rho)$ | Bedienzeit und Zwischenankunftszeit |
| $\theta \in (0,1]$ | **Extremalindex**: Kehrwert der mittleren Clustergröße im Grenzwert hoher Schwellen (nicht der Extremwertindex $\xi$) |
| $u$, $k$, $\zeta = k/n$ | Schwelle, Zahl der Überschreitungen, Anteil über der Schwelle |
| $x_p$ | Pegel zur Überschreitungswahrscheinlichkeit $p$ **je Kunde** |

**Modell.** Mit $W_1 = 0$ (Wahrscheinlichkeit $1-\rho$) oder $W_1 \sim \mathrm{Exp}(\gamma)$ (Wahrscheinlichkeit $\rho$), also stationär gestartet:

$$W_{n+1} = \max\left(0,\; W_n + S_n - A_{n+1}\right)$$

Die stationäre Verteilung ist exakt $P(W = 0) = 1-\rho$ und $P(W > x) = \rho\, e^{-\gamma x}$; der Pegel ist $x_p = \ln(\rho/p)/\gamma$, die Überschreitungen jeder Schwelle sind exponentiell ($\xi = 0$).

**Extremalindex.** Nach O'Brien ist

$$\theta = \lim_{r\to\infty}\lim_{u\to\infty} P\left(W_1,\dots,W_r \le u \mid W_0 > u\right)$$

Skizze: gegeben $W_0 > u$ ist der Überschuss $\mathrm{Exp}(\gamma)$ (gedächtnislos), die Zukunft eine Irrfahrt mit den Schritten $S - A$. Mit der Verweilzeit $T = S + W$, die im M/M/1 exakt $\mathrm{Exp}(\gamma)$ ist, und $V = T - A$ folgt

$$\theta = E\left[\,1 - e^{\gamma V};\; V < 0\,\right] = \gamma - \rho\gamma = (1-\rho)^2$$

Numerisch: Der Kern der Kette auf einem Gitter (Zustand 0 und Zellen der Breite $h$) hat bei $u$ den dominanten Eigenwert $\lambda_u$ seines abgetöteten Teils; dann ist $\theta(u) = (1-\lambda_u)/\bar F_h(u)$, und $\theta(u) \to (1-\rho)^2$ für $u \to \infty$.

**Tagesmaximum.** Mit dem Kern $Q$ auf den Zuständen bis $x$ und der stationären Verteilung $\pi$:

$$P\left(\max_{i \le n} W_i \le x\right) = \pi\, Q^{\,n-1}\, \mathbf{1}$$

Gumbel-Näherung mit $\theta$ (gültig erst bei vielen unabhängigen Clustern, $n\theta\rho \gg 1$) und Quantil:

$$P\left(\max_{i \le n} W_i \le x\right) \approx \exp\left(-n\,\theta\,\rho\, e^{-\gamma x}\right), \qquad x_q = \frac{\ln(n\theta\rho) - \ln(-\ln q)}{\gamma}$$

Die Annahme „unabhängig“ setzt $\theta = 1$; im Grenzwert verschiebt sie das Quantil um $\ln(1/\theta)/\gamma$ Abfertigungsdauern.

**Pegel und Intervall.** Die GPD$(\xi,\sigma)$ wird an die $k$ Überschreitungen angepasst (Maximum Likelihood); der Pegel zur Überschreitungswahrscheinlichkeit $p$ je Kunde ist

$$x_p = u + \frac{\sigma}{\xi}\left(\left(\frac{\zeta}{p}\right)^{\xi} - 1\right) \quad (\xi \ne 0), \qquad x_p = u + \sigma \ln\frac{\zeta}{p} \quad (\xi = 0)$$

Das Delta-Intervall für $\ln x_p$ nutzt die erwartete Information der GPD und die binomiale Varianz von $\zeta$; bei Abhängigkeit wird die gesamte Varianz durch $\theta$ geteilt (Breite mal $1/\sqrt{\theta}$), $k$ Überschreitungen zählen dann wie $k\theta$ Cluster. Das ist eine Konstruktion: in den Läufen streut der Pegel stärker, als dieses Teilen annimmt.

**Schätzer von $\theta$.** Runs: $\hat\theta = $ Zahl der Cluster / Zahl der Überschreitungen (neuer Cluster, wenn der Abstand größer als $r$ ist). Blöcke: mit $B$ Blöcken der Länge $b$, $K$ Blöcken mit Überschreitung und $N$ Überschreitungen

$$\hat\theta = \frac{\ln(1 - K/B)}{b\,\ln\left(1 - N/(Bb)\right)}$$

Intervalle: nach Ferro und Segers (2003), aus den Abständen der Überschreitungen (Formel aus dem Gedächtnis, an exakten Werten geprüft).

**max-AR(1).** Mit $X_t = \max(\varphi X_{t-1}, (1-\varphi) Z_t)$, $Z_t \sim$ Fréchet(1), ist $\theta = 1-\varphi$ und das Maximum über $b$ aufeinanderfolgende Werte exakt Fréchet mit der Skala $s_b = 1 + (b-1)(1-\varphi)$:

$$P(M_b \le x) = \exp\left(-\frac{s_b}{x}\right)$$

Der 100-Block-Wert $x^* = s_b/(-\ln 0{,}99)$ hat je Beobachtung die Überschreitungswahrscheinlichkeit $p^* = 1 - e^{-1/x^*}$; nur bei $\theta = 1$ ist $p^* = 1 - 0{,}99^{1/b}$.

**Literatur** (Stufe: W = in der Quellenliste der Planung als gelesen geführt, A = Zusammenfassung oder Kurzseite, G = aus dem Gedächtnis). Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*, Satz 5.2 und Gleichung (5.4), W. Embrechts, Klüppelberg, Mikosch (1997), *Modelling Extremal Events*, Definition 8.1.2, W.
Smith (1987) für die erwartete Information der GPD, G (das Vorzeichen des Nebenglieds war zuerst falsch, jetzt an einer Simulation geprüft). Ferro und Segers (2003) für den Intervall-Schätzer, G. Peshkova (2026), *Discrete Contin. Models Appl. Comput. Sci.* 34(1), 24–39, für θ = (1−ρ)² im M/M/1: A, Wortlaut nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html)."
)
