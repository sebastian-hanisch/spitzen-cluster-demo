"""Erzeugt und prüft `clu_coverage.json`: die eingefrorenen Läufe der Seite (Zufall: SplitMix64 als numpy-Vektor, `clu_queue.py`; Zähleinheit Läufe).

1. `abdeckung` (Experiment 3): Abdeckung des 95-%-Delta-Intervalls für den Pegel zur Überschreitungswahrscheinlichkeit p je Kunde der M/M/1-Wartezeit, Last ρ, n aufeinanderfolgende Kunden
   (stationär gestartet), Schwelle = empirisches (1 − Anteil)-Quantil, GPD-ML (ξ frei), Intervalle für ln x_p: (i) wie bei Unabhängigkeit, (ii) Varianz/θ̂ (Intervall-Schätzer), (iii) Varianz/θ exakt.
   Die GPD ist hier exakt (ξ = 0, jede Schwelle): ein Abdeckungsfehler kommt aus der Abhängigkeit und, bei wenigen Überschreitungen, aus der Näherung des Delta-Intervalls (an unabhängigen Daten deckt es in der Hauptzelle rund 89 %, `tests/test_oracle_gpd.py`). R Läufe je Zelle.
2. `schaetzer` (Experiment 1): θ̂ der drei Schätzer an max-AR(1) (exaktes θ = 1 − φ, n = 5000) und an der M/M/1-Wartezeit (θ(u) aus dem Kern).
3. `block_schwelle` (Experiment 5): Block gegen Schwelle unter Abhängigkeit am max-AR(1) (m = 50 Blöcke zu b = 50, Ziel der 100-Block-Wert x* = s_b/(−ln 0,99)); die Block-GEV nutzt scipy
   (Werkzeug, nie die App); die GPD wird beim θ-bereinigten Niveau p* = 1 − exp(−1/x*) gerechnet (die erste Fassung der Vormessung rechnete bei p = 1 − 0,99^(1/b): nur bei θ = 1 dasselbe Niveau).

Aufruf (Test-venv, im Ordner der Demo; OMP_NUM_THREADS=1):
    python tools/clu_make_coverage.py --write [--workers 5] [--only abdeckung|schaetzer|block_schwelle]
    python tools/clu_make_coverage.py --check      # rechnet die Zelle der Hauptaussage (Last 0,8; n = 3000) mit R = 300 neu und vergleicht als Band (vier Standardfehler)"""

import argparse
import json
import math
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import clu_evaluation as EV  # noqa: E402
import clu_kernel as K  # noqa: E402
import clu_pot as P  # noqa: E402
import clu_queue as Q  # noqa: E402
import clu_theta as TH  # noqa: E402

OUT = ROOT / "clu_coverage.json"
STAND = "2026-11-20"

# --- Abdeckung -------------------------------------------------------------------------------------------------------------------------------
RHOS = (0.5, 0.8)
NS = (1000, 3000, 12000)
FRACS = (0.10, 0.05, 0.02)
PS = (1e-3, 1e-4)
R_COV = 2000
SEED_COV = 7_100_000
CHUNK = 250

# --- Schätzer --------------------------------------------------------------------------------------------------------------------------------
PHIS = (0.0, 0.5, 0.8)
QS_EST = (0.95, 0.99)
N_EST, R_EST, SEED_EST = 5000, 500, 7_200_000
MM1_RHOS = (0.5, 0.8, 0.9)
MM1_FBARS = (1e-2, 1e-3)
N_MM1, R_MM1, SEED_MM1 = 100_000, 40, 7_300_000

# --- Block gegen Schwelle ----------------------------------------------------------------------------------------------------------------------
B_BLOCK, M_BLOCK = 50, 50
KS_H = (50, 125)
R_H, SEED_H = 1500, 7_400_000


def cell_seed(*idx) -> int:
    s = 0
    for i in idx:
        s = s * 101 + int(i)
    return s


# ------------------------------------------------------------------------------------------------------------------------------------------
def cov_chunk(args):
    """Läufe r0 bis r1 − 1 einer Zelle; dieselbe Rechnung wie die Live-Läufe der App (`clu_evaluation.coverage_rows`)."""
    rho, n, frac, p, seed, r0, r1 = args
    return EV.coverage_rows(rho, n, frac, p, seed, r0, r1)


summarise_cov = EV.summarise_coverage


def build_abdeckung(pool, rhos=RHOS, ns=NS, fracs=FRACS, ps=PS, R=R_COV):
    cells = []
    tasks = []
    for ir, rho in enumerate(rhos):
        for inn, n in enumerate(ns):
            for ifr, frac in enumerate(fracs):
                for ip, p in enumerate(ps):
                    seed = SEED_COV + cell_seed(ir, inn, ifr, ip)
                    cells.append((rho, n, frac, p, seed))
                    tasks += [(rho, n, frac, p, seed, r0, min(R, r0 + CHUNK)) for r0 in range(0, R, CHUNK)]
    results = pool.map(cov_chunk, tasks)
    per_cell = {}
    for t, rows in zip(tasks, results):
        per_cell.setdefault(t[:5], []).extend(rows)
    out = []
    for key in cells:
        rho, n, frac, p, seed = key
        d = {"rho": rho, "n": n, "anteil": frac, "p": p, "seed": seed, "x_wahr": Q.mm1_level(rho, p)}
        d.update(summarise_cov(per_cell[key], rho))
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------
def est_chunk(args):
    kind, a, q, seed, r0, r1 = args
    rows = []
    for run in range(r0, r1):
        if kind == "maxar":
            x = Q.maxar1(a, N_EST, seed, run)
            u = -1.0 / math.log(q)
        else:
            x = Q.mm1_wait(a, N_MM1, seed, run)
            u = Q.mm1_threshold(a, q)
        rows.append([TH.theta_runs(x, u, 10), TH.theta_blocks(x, u, 50), TH.theta_intervals(x, u), float(np.sum(x > u)), len(TH.clusters(x, u, 10))])
    return rows


def build_schaetzer(pool):
    tasks, keys = [], []
    for ip, phi in enumerate(PHIS):
        for iq, q in enumerate(QS_EST):
            seed = SEED_EST + cell_seed(ip, iq)
            keys.append(("maxar", phi, q, seed))
            tasks += [("maxar", phi, q, seed, r0, min(R_EST, r0 + 100)) for r0 in range(0, R_EST, 100)]
    for ir, rho in enumerate(MM1_RHOS):
        for ifb, fb in enumerate(MM1_FBARS):
            seed = SEED_MM1 + cell_seed(ir, ifb)
            keys.append(("mm1", rho, fb, seed))
            tasks += [("mm1", rho, fb, seed, r0, min(R_MM1, r0 + 10)) for r0 in range(0, R_MM1, 10)]
    results = pool.map(est_chunk, tasks)
    per = {}
    for t, rows in zip(tasks, results):
        per.setdefault((t[0], t[1], t[2], t[3]), []).extend(rows)
    out = []
    for key in keys:
        kind, a, q, seed = key
        arr = np.array(per[key], dtype=float)
        d = {"modell": kind, "seed": seed, "R": int(len(arr)), "k_median": float(np.median(arr[:, 3])), "cluster_runs_median": float(np.median(arr[:, 4]))}
        if kind == "maxar":
            d.update({"phi": a, "theta_exact": 1 - a, "quantil": q, "n": N_EST})
        else:
            d.update({"rho": a, "randanteil": q, "n": N_MM1, "theta_exact": (1 - a) ** 2})
            d["theta_kern_h02"] = K.theta_of_threshold(a, q, 0.2)
        for j, name in enumerate(("runs_r10", "bloecke_b50", "intervalle")):
            v = arr[:, j]
            v = v[np.isfinite(v)]
            d[name] = {"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)), "n": int(len(v))}
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------
def gev_nll(t, x):
    mu, ls, xi = t
    if xi < -1.0:
        return 1e300
    z = (x - mu) * math.exp(-ls)
    if abs(xi) < 1e-7:
        return x.size * ls + float(np.sum(z) + np.sum(np.exp(-z)))
    w = 1 + xi * z
    if np.any(w <= 0):
        return 1e300
    lw = np.log(w)
    return x.size * ls + (1 + 1 / xi) * float(np.sum(lw)) + float(np.sum(np.exp(-lw / xi)))


def gev_fit(x):
    s0 = float(x.std(ddof=1)) * math.sqrt(6) / math.pi
    best = None
    for mu0, ls0, xi0 in ((float(np.median(x)) - 0.4 * s0, math.log(s0), 0.1), (float(np.quantile(x, 0.4)), math.log(max(float(np.subtract(*np.quantile(x, [0.75, 0.25]))) / 2, 1e-9)), 0.6)):
        p0 = np.array([mu0, ls0, xi0])
        sc = np.array([x.std(), 0.1, 0.1])
        sim = np.vstack([p0] + [p0 + sc * 0.5 * np.eye(3)[i] for i in range(3)])
        r = optimize.minimize(gev_nll, p0, args=(x,), method="Nelder-Mead", options={"initial_simplex": sim, "xatol": 1e-5, "fatol": 1e-8, "maxiter": 2000, "maxfev": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun)


def gev_level(t, pexc):
    mu, ls, xi = t
    y = -math.log1p(-pexc)
    if abs(xi) < 1e-9:
        return mu - math.exp(ls) * math.log(y)
    return mu + math.exp(ls) * (y ** (-xi) - 1) / xi


def gev_delta(x, t, pexc):
    """Delta-Intervall mit beobachteter Information (numerische Hesse-Matrix); (z, se) oder None, wenn die Matrix nicht positiv definit ist."""
    t = np.asarray(t, float)
    h = np.array([1e-4 * math.exp(t[1]), 1e-4, 1e-4])
    H = np.zeros((3, 3))
    f0 = gev_nll(t, x)
    for i in range(3):
        for j in range(i, 3):
            ei, ej = np.zeros(3), np.zeros(3)
            ei[i], ej[j] = h[i], h[j]
            if i == j:
                H[i, i] = (gev_nll(t + ei, x) - 2 * f0 + gev_nll(t - ei, x)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (gev_nll(t + ei + ej, x) - gev_nll(t + ei - ej, x) - gev_nll(t - ei + ej, x) + gev_nll(t - ei - ej, x)) / (4 * h[i] * h[j])
    try:
        if not np.all(np.isfinite(H)) or np.any(np.linalg.eigvalsh(H) <= 0):
            return None
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h[i]
        g[i] = (gev_level(t + e, pexc) - gev_level(t - e, pexc)) / (2 * h[i])
    var = float(g @ cov @ g)
    return None if var <= 0 else (gev_level(t, pexc), math.sqrt(var))


def h_chunk(args):
    phi, seed, r0, r1 = args
    N = B_BLOCK * M_BLOCK
    theta = 1 - phi
    sb = Q.maxar1_block_scale(phi, B_BLOCK)
    xs = sb / (-math.log(0.99))
    pstar = 1 - math.exp(-1 / xs)
    p_obs = 1 - 0.99 ** (1 / B_BLOCK)
    lx = math.log(xs)
    rows = []
    for run in range(r0, r1):
        x = Q.maxar1(phi, N, seed, run)
        mx = x.reshape(M_BLOCK, B_BLOCK).max(axis=1)
        t, _f = gev_fit(mx)
        d = gev_delta(mx, t, 0.01)
        if d is None:
            blk = [math.nan, math.nan, gev_level(t, 0.01) / xs]
        else:
            z, se = d
            blk = [float(z - 1.959964 * se <= xs <= z + 1.959964 * se), float(abs(lx - math.log(z)) <= 1.959964 * se / z), z / xs]
        row = list(blk)
        for k in KS_H:
            u, kk, y = P.fit_topk(x, k)
            xi, sg, _e = P.gpd_ml(y)
            if not math.isfinite(xi):
                row += [math.nan] * 5
                continue
            lo1, hi1 = P.delta_ci_logx(xi, sg, u, k, N, p_obs, theta)         # alte Fassung: Niveau p_obs, exaktes θ
            lo2, hi2 = P.delta_ci_logx(xi, sg, u, k, N, pstar, theta)         # θ-bereinigtes Niveau, exaktes θ
            lo3, hi3 = P.delta_ci_logx(xi, sg, u, k, N, pstar, 1.0)           # θ-bereinigtes Niveau, wie bei Unabhängigkeit
            lev2 = P.pot_level(xi, sg, u, k / N, pstar)
            row += [float(lo1 <= lx <= hi1), float(lo2 <= lx <= hi2), float(lo3 <= lx <= hi3), lev2 / xs, P.pot_level(xi, sg, u, k / N, p_obs) / xs]
        rows.append(row)
    return rows


def build_block_schwelle(pool):
    tasks, keys = [], []
    for ip, phi in enumerate(PHIS):
        seed = SEED_H + cell_seed(ip)
        keys.append((phi, seed))
        tasks += [(phi, seed, r0, min(R_H, r0 + 100)) for r0 in range(0, R_H, 100)]
    results = pool.map(h_chunk, tasks)
    per = {}
    for t, rows in zip(tasks, results):
        per.setdefault((t[0], t[1]), []).extend(rows)
    out = []

    def cov(col):
        c = col[np.isfinite(col)]
        m = float(np.mean(c))
        return {"abdeckung": m, "se": math.sqrt(max(m * (1 - m), 1e-12) / len(c)), "n": int(len(c))}

    for phi, seed in keys:
        a = np.array(per[(phi, seed)], dtype=float)
        theta = 1 - phi
        sb = Q.maxar1_block_scale(phi, B_BLOCK)
        xs = sb / (-math.log(0.99))
        d = {"phi": phi, "theta": theta, "seed": seed, "R": int(len(a)), "x_ziel": xs, "p_stern": 1 - math.exp(-1 / xs), "p_alt": 1 - 0.99 ** (1 / B_BLOCK),
             "block_gev_pegelskala": cov(a[:, 0]), "block_gev_lnskala": cov(a[:, 1]), "block_verhaeltnis_median": float(np.median(a[:, 2])), "block_anteil_unter": float(np.mean(a[:, 2] < 1))}
        for i, k in enumerate(KS_H):
            o = 3 + 5 * i
            d[f"k{k}"] = {
                "cluster": k * theta,
                "alt_theta_exakt": cov(a[:, o]), "bereinigt_theta_exakt": cov(a[:, o + 1]), "bereinigt_theta_1": cov(a[:, o + 2]),
                "verhaeltnis_bereinigt_median": float(np.nanmedian(a[:, o + 3])), "verhaeltnis_alt_median": float(np.nanmedian(a[:, o + 4])),
            }
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------
def build(workers, only=None):
    t0 = time.time()
    data = {}
    with Pool(workers) as pool:
        if only in (None, "abdeckung"):
            data["abdeckung"] = build_abdeckung(pool)
        if only in (None, "schaetzer"):
            data["schaetzer"] = build_schaetzer(pool)
        if only in (None, "block_schwelle"):
            data["block_schwelle"] = build_block_schwelle(pool)
    meta = {
        "quelle": "tools/clu_make_coverage.py (SplitMix64 als numpy-Vektor, clu_queue.py); eigene Messung dieser Demo, Zähleinheit Läufe",
        "stand": STAND, "software": {"python": platform.python_version(), "numpy": np.__version__},
        "parameter": {"R_abdeckung": R_COV, "R_schaetzer_maxar": R_EST, "R_schaetzer_mm1": R_MM1, "R_block_schwelle": R_H, "n_schaetzer_maxar": N_EST, "n_schaetzer_mm1": N_MM1,
                      "b_block": B_BLOCK, "m_block": M_BLOCK, "k_block_schwelle": list(KS_H)},
        "rechenzeit_s": round(time.time() - t0, 1),
    }
    return meta, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", choices=("abdeckung", "schaetzer", "block_schwelle"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.write:
        old = json.loads(OUT.read_text(encoding="utf-8")) if (OUT.exists() and a.only) else {}
        meta, data = build(a.workers, a.only)
        old.update(meta)
        old.update(data)
        OUT.write_text(json.dumps(old, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"geschrieben: {OUT.name}, {OUT.stat().st_size / 1024:.0f} KB, {meta['rechenzeit_s']} s")
    elif a.check:
        old = json.loads(OUT.read_text(encoding="utf-8"))["abdeckung"]
        cell = next(c for c in old if c["rho"] == 0.8 and c["n"] == 3000 and c["anteil"] == 0.05 and c["p"] == 1e-3)
        with Pool(a.workers) as pool:
            rows = pool.map(cov_chunk, [(0.8, 3000, 0.05, 1e-3, cell["seed"], r0, r0 + 100) for r0 in range(0, 300, 100)])
        new = summarise_cov([r for part in rows for r in part], 0.8)
        bad = []
        for name in ("iid", "theta_hat", "theta_exact"):
            se = math.hypot(cell["se_" + name], new["se_" + name])
            diff = abs(cell["abdeckung_" + name] - new["abdeckung_" + name])
            print(f"  {name}: Datei {cell['abdeckung_' + name]:.3f}, neu (R = 300) {new['abdeckung_' + name]:.3f}, Abstand {diff:.3f}, vier Standardfehler {4 * se:.3f}")
            if diff > 4 * se:
                bad.append(name)
        sys.exit(1 if bad else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
