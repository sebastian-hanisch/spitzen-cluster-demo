"""Auswertungen und Verdikte der Seite. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`.

Zähleinheiten stehen je Zahl im Text: Kunden, Überschreitungen, Cluster, Läufe, Blöcke. θ ist der Extremalindex, ξ der Extremwertindex.
Die Verdikte lesen ihre Zahlen aus der Rechnung dieser Sitzung oder aus den eingefrorenen Messungen (`clu_reference`), nie aus einem Text."""

import math
from dataclasses import dataclass, fields

import numpy as np

import clu_constants as C
import clu_kernel as K
import clu_pot as P
import clu_queue as Q
import clu_reference as REF
import clu_theta as TH
from clu_texts import de, de_signed, frac_text, minutes, p_text, pct, q_text, rho_text, thousands


@dataclass(frozen=True)
class Settings:
    rho1: float = C.DEFAULT_RHO1
    n1: int = C.DEFAULT_N1
    frac1: float = C.DEFAULT_FRAC1
    r1: int = C.DEFAULT_R1
    b1: int = C.DEFAULT_B1
    rho2: float = C.DEFAULT_RHO2
    n2: int = C.DEFAULT_N2
    q2: float = C.DEFAULT_Q2
    rho3: float = C.DEFAULT_RHO3
    n3: int = C.DEFAULT_N3
    frac3: float = C.DEFAULT_FRAC3
    p3: float = C.DEFAULT_P3
    r3: int = C.DEFAULT_R3
    rho4: float = C.DEFAULT_RHO4
    n4: int = C.DEFAULT_N4
    phi5: float = C.DEFAULT_PHI5
    k5: int = C.DEFAULT_K5
    seed: int = C.DEFAULT_SEED


SETTING_NAMES = tuple(f.name for f in fields(Settings))


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: eine Reihe, ihre Spitzen, ihre Cluster
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SeriesAnalysis:
    rho: float
    n: int
    f_bar: float
    r: int
    b: int
    x: np.ndarray
    u: float
    k: int
    clusters: tuple
    theta_runs: float
    theta_blocks: float
    theta_intervals: float
    theta_exact: float
    theta_kernel: float
    mean_cluster_exact: float
    clusters_expected: float

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)


def analyse_series(rho: float, n: int, f_bar: float, r: int, b: int, seed: int) -> SeriesAnalysis:
    """Eine stationär gestartete M/M/1-Reihe mit n Kunden; die Schwelle u hat den exakten Randanteil F̄ (P(W > u) = F̄)."""
    x = Q.mm1_wait(rho, n, seed)
    u = Q.mm1_threshold(rho, f_bar)
    cl = TH.clusters(x, u, r)
    k = int(np.sum(x > u))
    th = K.theta_exact(rho)
    return SeriesAnalysis(
        rho=rho, n=n, f_bar=f_bar, r=r, b=b, x=x, u=u, k=k, clusters=tuple(cl),
        theta_runs=TH.theta_runs(x, u, r), theta_blocks=TH.theta_blocks(x, u, b), theta_intervals=TH.theta_intervals(x, u),
        theta_exact=th, theta_kernel=K.theta_of_threshold(rho, f_bar, 0.2), mean_cluster_exact=1.0 / th, clusters_expected=K.clusters_among(k, th),
    )


def verdict_series(a: SeriesAnalysis):
    if a.k == 0:
        return "warning", (f"Last {rho_text(a.rho)}: in dieser Reihe liegt kein Kunde über der Schwelle (n = {thousands(a.n)} Kunden, Anteil {frac_text(a.f_bar)}). "
                           "Ohne Überschreitung gibt es nichts zu schätzen; mehr Kunden oder ein größerer Anteil wählen.")
    text = (f"Last {rho_text(a.rho)}: exakt θ = {de(a.theta_exact, 4)} (Grenzwert hoher Schwellen), also im Mittel {de(a.mean_cluster_exact, 0)} Überschreitungen je Cluster (je Spitze); "
            f"bei dieser Schwelle liefert der Kern θ(u) = {de(a.theta_kernel, 4)}. Unter den {thousands(a.k)} Überschreitungen dieser Reihe stecken rechnerisch nur "
            f"k·θ = {de(a.clusters_expected, 1)} unabhängige Cluster; die Runs-Regel (r = {a.r}) zählt hier {a.n_clusters} Cluster. "
            f"θ̂ in dieser einen Reihe: Runs {de(a.theta_runs, 3)}, Blöcke (b = {a.b}) {de(a.theta_blocks, 3)}, Intervalle {de(a.theta_intervals, 3)}. "
            "Eine Reihe ist kein Befund: die Tabelle unten zeigt den Median über 500 Läufe.")
    kind = "warning" if a.clusters_expected < C.MIN_CLUSTERS_OK else "info"
    return kind, text


def estimator_rows(model: str):
    return [r for r in REF.estimator_rows() if r["modell"] == model]


def verdict_estimators(rows: list):
    """Verdikt über die eingefrorenen Schätzer-Zeilen des max-AR(1) (exaktes θ = 1 − φ, bei φ = 0 unabhängige Daten)."""
    iid = next(r for r in rows if r["modell"] == "maxar" and r["phi"] == 0.0 and r["quantil"] == 0.95)
    dep = next(r for r in rows if r["modell"] == "maxar" and r["phi"] == 0.8 and r["quantil"] == 0.95)
    text = (f"An unabhängigen Daten (max-AR(1) mit φ = 0, {thousands(iid['n'])} Beobachtungen, 5 % über der Schwelle, {iid['R']} Läufe) liefert der Runs-Schätzer mit festem r = 10 im Median "
            f"{de(iid['runs_r10']['median'], 2)} statt 1,0, die Blöcke (b = 50) {de(iid['bloecke_b50']['median'], 2)}, die Intervalle {de(iid['intervalle']['median'], 2)}: "
            f"zwei Überschreitungen zählen als verschiedene Cluster, sobald ihr Abstand größer als 10 Kunden ist, und bei 5 % Überschreitungen kommt das ohne jede Abhängigkeit vor. "
            f"Bei φ = 0,8 (θ = 0,2) sind die Mediane {de(dep['runs_r10']['median'], 2)} / {de(dep['bloecke_b50']['median'], 2)} / {de(dep['intervalle']['median'], 2)}. "
            "Die Schätzer hängen an r, b und der Schwelle; gezählt wird in Clustern, nicht in Überschreitungen.")
    return "info", text


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: das Tagesmaximum
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DayView:
    rho: float
    n: int
    q: float
    xs: np.ndarray
    exact: np.ndarray
    gumbel_theta: np.ndarray
    gumbel_iid: np.ndarray
    q_exact: float
    q_theta: float
    q_iid: float
    shift: float
    shift_limit: float
    independent_clusters: float      # n·θ·ρ


def day_view(rho: float, n: int, q: float) -> DayView:
    xs, cdf = REF.day_cdf(rho, n)
    th = K.theta_exact(rho)
    qe = REF.day_quantile(rho, n, q)
    q_theta = K.gumbel_day_quantile(rho, n, q)
    q_iid = K.gumbel_day_quantile(rho, n, q, theta=1.0)
    return DayView(rho=rho, n=n, q=q, xs=xs, exact=cdf, gumbel_theta=K.gumbel_day_cdf(rho, n, xs), gumbel_iid=K.gumbel_day_cdf(rho, n, xs, theta=1.0),
                   q_exact=qe, q_theta=q_theta, q_iid=q_iid, shift=q_iid - qe, shift_limit=K.gumbel_shift(rho), independent_clusters=n * th * rho)


def verdict_day(v: DayView):
    dev = (v.q_theta - v.q_exact) / v.q_exact
    approx_ok = abs(dev) <= 0.10
    text = (f"Last {rho_text(v.rho)}, {thousands(v.n)} Kunden, {q_text(v.q)}-Quantil des Tagesmaximums (Abfertigungsdauern): exakt {de(v.q_exact, 1)}, "
            f"als wären die Kunden unabhängig gerechnet {de(v.q_iid, 1)}: Verschiebung {de(v.shift, 1)} Abfertigungsdauern (bei 3 min je Abfertigung ≈ {minutes(v.shift)}). "
            f"Der Grenzwert ln(1/θ)/γ ist {de(v.shift_limit, 1)} (≈ {minutes(v.shift_limit)}), nicht dieselbe Zahl: er gilt für unendlich viele Kunden. "
            f"Gumbel mit θ = (1−ρ)²: {de(v.q_theta, 1)} ({'+' if dev >= 0 else '−'}{de(abs(dev) * 100, 0)} % gegen exakt). "
            f"Die Näherung braucht viele unabhängige Cluster je Tag; hier sind es n·θ·ρ = {de(v.independent_clusters, 1)}.")
    return ("info" if approx_ok else "warning"), text


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: hält das 95-%-Intervall?
# ------------------------------------------------------------------------------------------------------------------------------------------
def coverage_rows(rho: float, n: int, frac: float, p: float, seed: int, r0: int, r1: int) -> list:
    """Läufe r0 bis r1 − 1 der Abdeckungsstudie (eingefroren im Werkzeug, live in der App: dieselbe Rechnung). Zeile je Lauf: [Fehler ln, k, θ̂, Cluster, (Abdeckung, Breite, 0) × 3]."""
    xt = math.log(Q.mm1_level(rho, p))
    th_exact = K.theta_exact(rho)
    out = []
    for run in range(r0, r1):
        x = Q.mm1_wait(rho, n, seed, run)
        u, k, y = P.fit_threshold(x, frac)
        if k < 4:
            out.append(None)
            continue
        xi, sg, _edge = P.gpd_ml(y)
        if not math.isfinite(xi):
            out.append(None)
            continue
        lev = P.pot_level(xi, sg, u, k / n, p)
        th_hat = TH.theta_intervals(x, u)
        row = [math.log(lev) - xt, k, th_hat, len(TH.clusters(x, u, TH.R_DEFAULT))]
        for thv in (1.0, th_hat, th_exact):
            if not math.isfinite(thv):
                row += [math.nan, math.nan, math.nan]
                continue
            lo, hi = P.delta_ci_logx(xi, sg, u, k, n, p, thv)
            row += [float(lo <= xt <= hi), hi - lo, 0.0]
        out.append(row)
    return out


def summarise_coverage(rows: list, rho: float) -> dict:
    a = np.array([r for r in rows if r is not None], dtype=float)
    res = {"R": int(len(a)), "theta_exact": K.theta_exact(rho)}
    for name, col in (("iid", 4), ("theta_hat", 7), ("theta_exact", 10)):
        c = a[:, col]
        ok = np.isfinite(c)
        cov = float(np.mean(c[ok]))
        res["abdeckung_" + name] = cov
        res["se_" + name] = math.sqrt(max(cov * (1 - cov), 1e-12) / int(ok.sum()))
        res["n_" + name] = int(ok.sum())
        res["breite_" + name] = float(np.median(a[ok, col + 1]))
    err = a[:, 0]
    res["verzerrung_ln"] = float(err.mean())
    res["sd_ln"] = float(err.std(ddof=1)) if len(err) > 1 else math.nan
    res["anteil_unter_wahr"] = float(np.mean(err < 0))
    res["k_median"] = float(np.median(a[:, 1]))
    res["theta_hat_median"] = float(np.nanmedian(a[:, 2])) if np.any(np.isfinite(a[:, 2])) else math.nan
    res["cluster_runs_median"] = float(np.median(a[:, 3]))
    return res


def live_coverage(rho: float, n: int, frac: float, p: float, R: int, seed: int) -> dict | None:
    """R Läufe live (Zufalls-Seed der Seitenleiste); None bei R = 0. Dieselbe Rechnung wie in der eingefrorenen Studie, andere Reihen."""
    if R <= 0:
        return None
    return summarise_coverage(coverage_rows(rho, n, frac, p, seed, 0, R), rho)


def verdict_coverage(c: dict, rho: float, n: int, frac: float, p: float):
    k = c["k_median"]
    th = c["theta_exact"]
    text = (f"Last {rho_text(rho)}, {thousands(n)} Kunden, Schwelle bei {frac_text(frac)} der Kunden (im Median {thousands(k)} Überschreitungen), Pegel zu p = {p_text(p)} je Kunde, "
            f"{thousands(c['R'])} Läufe: das 95-%-Intervall mit der Annahme „unabhängig“ deckt den wahren Pegel in {pct(c['abdeckung_iid'])} (Standardfehler {de(100 * c['se_iid'], 1)} Punkte), "
            f"mit geschätztem θ in {pct(c['abdeckung_theta_hat'])}, mit dem exakten θ = {de(th, 3)} in {pct(c['abdeckung_theta_exact'])}. "
            f"Unter {thousands(k)} Überschreitungen stecken rechnerisch k·θ = {de(k * th, 1)} unabhängige Cluster. Auch die Korrektur mit dem richtigen θ erreicht in der Studie (36 Zellen) nirgends 93 %: die Streuung des Pegels ist größer, als das Teilen der Varianz durch θ unterstellt "
            f"(Standardabweichung von ln des Pegels {de(c['sd_ln'], 3)} gegen {de(c['breite_theta_exact'] / (2 * P.Z95), 3)} im Intervall mit exaktem θ).")
    if c["abdeckung_theta_exact"] >= 0.93:
        kind = "success" if c["abdeckung_iid"] >= 0.93 else "info"
    else:
        kind = "warning"
    return kind, text


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: wie groß muss der Block sein?
# ------------------------------------------------------------------------------------------------------------------------------------------
def block_summary(rho: float):
    """Zeilen (n, KS-ξ, ML-ξ, KS-Abstand GEV, KS-Abstand Gumbel, σ) für alle eingefrorenen Blocklängen."""
    rows = []
    for n in C.N4_OPTIONS:
        d = REF.block_row(rho, n)
        rows.append({"n": n, "xi_ks": d["ks_gev"]["xi"], "xi_ml": d["ml_gev"]["xi"], "ks_gev": d["ks_gev"]["ks"], "ks_gumbel": d["ks_gumbel"]["ks"],
                     "sigma_ks": d["ks_gev"]["sigma"], "sigma_limit": d["sigma_grenzwert"]})
    return rows


def verdict_blocks(rho: float, n: int):
    rows = block_summary(rho)
    sel = next(r for r in rows if r["n"] == n)
    xs = " / ".join(de_signed(r["xi_ks"], 3) for r in rows)
    ns = " / ".join(thousands(r["n"]) for r in rows)
    text = (f"Last {rho_text(rho)}: bei {thousands(n)} Kunden je Block hat die beste GEV (kleinster Kolmogorov-Abstand) ξ = {de_signed(sel['xi_ks'], 3)}, "
            f"obwohl der Grenzwert ξ = 0 ist (Wartezeit mit exponentiellem Rand). Über {ns} Kunden je Block: ξ = {xs}. "
            f"Die Skala σ = {de(sel['sigma_ks'], 2)} liegt unter dem Grenzwert 1/γ = {de(sel['sigma_limit'], 1)}. Beste Gumbel-Anpassung: Kolmogorov-Abstand {de(sel['ks_gumbel'], 4)} gegen {de(sel['ks_gev'], 4)} der GEV. "
            "Das ist ein Populationseffekt der Blocklänge, kein Zufall: der Populationswert der Likelihood-Anpassung liegt bei dieser Blocklänge über null (eine Anpassung an eine Stichprobe von Tagesmaxima streut um ihn).")
    return "info", text


# ------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: Block gegen Schwelle unter Abhängigkeit
# ------------------------------------------------------------------------------------------------------------------------------------------
def verdict_block_schwelle(row: dict, k: int):
    kk = row[f"k{k}"]
    th = row["theta"]
    blk, bln = row["block_gev_pegelskala"], row["block_gev_lnskala"]
    text = (f"max-AR(1) mit φ = {de(row['phi'], 1)} (θ = {de(th, 1)}), 50 Blöcke zu 50 Beobachtungen, Ziel der 100-Block-Wert, {thousands(row['R'])} Läufe: die Block-GEV deckt in {pct(blk['abdeckung'])} "
            f"(Delta-Intervall auf der Pegelskala; ln-Skala {pct(bln['abdeckung'])}). Der Schwellenweg mit {k} Überschreitungen (rechnerisch {de(kk['cluster'], 1)} Cluster) deckt ohne θ in "
            f"{pct(kk['bereinigt_theta_1']['abdeckung'])}, mit dem exakten θ in {pct(kk['bereinigt_theta_exakt']['abdeckung'])}. "
            "Das Niveau ist dabei θ-bereinigt (p* statt 1 − 0,99^(1/b)); der Blockweg verliert durch die Abhängigkeit weniger als der Schwellenweg (andere Blocklängen als 50 sind nicht gemessen).")
    kind = "success" if kk["bereinigt_theta_exakt"]["abdeckung"] >= 0.93 else "warning"
    return kind, text
