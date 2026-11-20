"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import clu_constants as C
import clu_kernel as K
from clu_texts import de, frac_text, rho_text


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_series(a):
    """Die Wartezeit-Reihe: graue Linie, Überschreitungen der Schwelle farbig nach Cluster (Runs-Regel), Schwelle gestrichelt."""
    n = a.x.size
    idx = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=a.x, mode="lines", line=dict(color=C.COLOR_GREY, width=1), name="Wartezeit", hovertemplate="Kunde %{x}: %{y:.2f}<extra></extra>"))
    if a.k:
        pos = np.concatenate(a.clusters)
        col = [C.CLUSTER_COLORS[j % len(C.CLUSTER_COLORS)] for j, cl in enumerate(a.clusters) for _ in cl]
        nr = [j + 1 for j, cl in enumerate(a.clusters) for _ in cl]
        fig.add_trace(go.Scatter(x=pos + 1, y=a.x[pos], mode="markers", marker=dict(color=col, size=5), showlegend=False, customdata=nr,
                                 hovertemplate="Cluster %{customdata}, Kunde %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1, n], y=[a.u, a.u], mode="lines", line=dict(color="gray", dash="dash"), name=f"Schwelle u = {de(a.u, 1)} (Anteil {frac_text(a.f_bar)})", hoverinfo="skip"))
    fig.update_xaxes(title_text="Kunde Nr.")
    fig.update_yaxes(title_text="Wartezeit (Abfertigungsdauern)", rangemode="tozero")
    return _base(fig, 340)


def build_day_cdf(v):
    """Verteilungsfunktion des Tagesmaximums: exakt (Kern), Gumbel mit θ = (1−ρ)² und Gumbel mit θ = 1; senkrechte Linie am gewählten Quantil."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v.xs, y=v.exact, mode="lines", line=dict(color=C.COLOR_EXACT, width=4), name="exakt (Gitterkern)", hovertemplate="x = %{x:.1f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=v.xs, y=v.gumbel_theta, mode="lines", line=dict(color=C.COLOR_THETA, width=2, dash="dash"), name="Gumbel mit θ = (1−ρ)²", hovertemplate="x = %{x:.1f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=v.xs, y=v.gumbel_iid, mode="lines", line=dict(color=C.COLOR_IID, width=2, dash="dot"), name="Gumbel mit θ = 1 („unabhängig“)", hovertemplate="x = %{x:.1f}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=v.q, line=dict(color="gray", width=1))
    fig.update_xaxes(title_text="Tagesmaximum x (Abfertigungsdauern)", range=[float(v.xs[0]), float(v.xs[-1])])
    fig.update_yaxes(title_text="P(Tagesmaximum ≤ x)", range=[0, 1.02])
    return _base(fig)


def build_coverage(c, live=None):
    """Abdeckung der drei Intervallarten mit ±1 Standardfehler (eingefroren) und, wenn vorhanden, die Live-Läufe daneben."""
    names = ["unabhängig gerechnet", "mit geschätztem θ", "mit exaktem θ"]
    keys = ["iid", "theta_hat", "theta_exact"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[100 * c["abdeckung_" + k] for k in keys], marker_color=[C.COLOR_IID, C.COLOR_THETA, C.COLOR_EXACT],
                         error_y=dict(type="data", array=[100 * c["se_" + k] for k in keys], visible=True), name=f"eingefroren ({c['R']:,} Läufe)".replace(",", "."),
                         hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    if live is not None:
        fig.add_trace(go.Bar(x=names, y=[100 * live["abdeckung_" + k] for k in keys], marker_color="rgba(120,120,120,0.55)",
                             error_y=dict(type="data", array=[100 * live["se_" + k] for k in keys], visible=True), name=f"live ({live['R']} Läufe)", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=95, line=dict(color="gray", dash="dash"), annotation_text="nominell 95 %", annotation_position="top left")
    fig.update_yaxes(title_text="Abdeckung (% der Läufe)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _base(fig, 340)


def build_blocks(rho: float, n: int, row):
    """Exakte Verteilung des Blockmaximums (Punkte), beste GEV nach Kolmogorov-Abstand (Linie) und beste Gumbel-Verteilung (gestrichelt)."""
    xs = np.array(row["x"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=row["cdf"], mode="markers", marker=dict(color=C.COLOR_EXACT, size=6), name="exakt (Gitterkern)", hovertemplate="x = %{x:.1f}: %{y:.4f}<extra></extra>"))
    g = row["ks_gev"]
    grid = np.linspace(xs[0], xs[-1], 200)
    z = (grid - g["mu"]) / g["sigma"]
    t = 1 + g["xi"] * z
    cdf = np.where(t > 0, np.exp(-np.power(np.maximum(t, 1e-300), -1 / g["xi"])), 0.0)
    fig.add_trace(go.Scatter(x=grid, y=cdf, mode="lines", line=dict(color=C.COLOR_THETA, width=3), name=f"beste GEV (ξ = {de(g['xi'], 3)})", hoverinfo="skip"))
    gg = row["ks_gumbel"]
    fig.add_trace(go.Scatter(x=grid, y=np.exp(-np.exp(-(grid - gg["mu"]) / gg["sigma"])), mode="lines", line=dict(color=C.COLOR_IID, width=2, dash="dash"), name="beste Gumbel-Verteilung (ξ = 0)", hoverinfo="skip"))
    fig.update_xaxes(title_text=f"Blockmaximum bei {n:,} Kunden (Abfertigungsdauern)".replace(",", "."))
    fig.update_yaxes(title_text="P(Blockmaximum ≤ x)", range=[0, 1.02])
    return _base(fig)


def build_blocks_xi(rows, rho: float):
    """ξ der besten GEV (Kolmogorov-Abstand und Likelihood) gegen die Blocklänge."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["xi_ks"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_THETA, width=3), name="beste GEV nach Kolmogorov-Abstand", hovertemplate="%{x} Kunden: ξ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns, y=[r["xi_ml"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_EXACT, width=2, dash="dash"), name="beste GEV nach Likelihood", hovertemplate="%{x} Kunden: ξ = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="gray"), annotation_text="Grenzwert ξ = 0", annotation_position="bottom right")
    fig.update_xaxes(type="log", tickvals=ns, ticktext=[f"{n:,}".replace(",", ".") for n in ns], title_text="Kunden je Block")
    fig.update_yaxes(title_text=f"ξ der Anpassung (Last {rho_text(rho)})", rangemode="tozero")
    return _base(fig, 320)


def build_block_schwelle(row, k: int):
    """Abdeckung bei Abhängigkeit: Block-GEV (Pegelskala, ln-Skala) und Schwellenweg (ohne θ, mit exaktem θ)."""
    kk = row[f"k{k}"]
    names = ["Block-GEV (Pegelskala)", "Block-GEV (ln-Skala)", f"Schwelle, k = {k}, ohne θ", f"Schwelle, k = {k}, exaktes θ"]
    vals = [row["block_gev_pegelskala"], row["block_gev_lnskala"], kk["bereinigt_theta_1"], kk["bereinigt_theta_exakt"]]
    fig = go.Figure(go.Bar(x=names, y=[100 * v["abdeckung"] for v in vals], marker_color=[C.COLOR_GREY, C.COLOR_GREY, C.COLOR_IID, C.COLOR_EXACT],
                           error_y=dict(type="data", array=[100 * v["se"] for v in vals], visible=True), hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=95, line=dict(color="gray", dash="dash"), annotation_text="nominell 95 %", annotation_position="top left")
    fig.update_yaxes(title_text="Abdeckung (% der Läufe)", range=[0, 105])
    return _base(fig, 320)


def build_theta_vs_load():
    """θ = (1−ρ)² gegen die Last (geschlossene Formel) mit den mittleren Clustergrößen an drei Stellen."""
    rho = np.linspace(0.3, 0.9, 61)
    fig = go.Figure(go.Scatter(x=rho, y=[K.theta_exact(r) for r in rho], mode="lines", line=dict(color=C.COLOR_EXACT, width=3), hovertemplate="Last %{x:.2f}: θ = %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Last ρ")
    fig.update_yaxes(title_text="Extremalindex θ", type="log")
    return _base(fig, 260)
