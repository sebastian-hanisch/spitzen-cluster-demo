"""Erzeugt und prüft `clu_blocks.json`: der Populations-ξ der Blockmaxima der M/M/1-Wartezeit (Experiment 4). Kein Zufall.

Blockmaximum = größte Wartezeit unter n aufeinanderfolgenden Kunden (stationär gestartet), exakte Verteilung aus dem Gitterkern (h = 0,1) auf dem Raster der Breite 0,5. Daran werden angepasst:
(a) die GEV mit dem kleinsten Kolmogorov-Abstand (ξ frei) und die beste Gumbel-Verteilung (ξ = 0) als Vergleich, über die Rasterpunkte mit 0,01 ≤ F ≤ 0,999;
(b) die GEV, die den Erwartungswert der Log-Likelihood unter der exakten Verteilung maximiert (Zellenmitte, Zellenmasse aus F): der Wert, gegen den die Maximum-Likelihood-Anpassung an unendlich viele Blöcke läuft.
Beides ist ein Populationswert, kein Schätzer an Stichproben. Die Anpassung nutzt scipy (Werkzeug, nie die App).
Konvention: ξ > 0 schwerer Rand (Fréchet-Typ); `scipy.stats.genextreme` hat c = −ξ.

Aufruf (Test-venv, im Ordner der Demo; OMP_NUM_THREADS=1):
    python tools/clu_make_blocks.py --write [--workers 5]
    python tools/clu_make_blocks.py --check       # rechnet Last 0,5, n = 64 neu und vergleicht"""

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

import clu_kernel as K  # noqa: E402

OUT = ROOT / "clu_blocks.json"
STAND = "2026-11-20"
RHOS = (0.5, 0.8)
NS = (64, 256, 1024, 4096)
H = 0.1
STEP = 0.5
_GRIDS: dict = {}


def _grid(rho, n):
    key = (rho,)
    if key not in _GRIDS:
        hi = K.gumbel_day_quantile(rho, max(NS), 1 - 1e-5, theta=1.0) + 12.0
        _GRIDS[key] = K.Grid(rho, H, hi)
    return _GRIDS[key]


def task(args):
    rho, n, x = args
    return rho, n, x, _grid(rho, n).day_max_cdf(x, n)


def gev_cdf(x, mu, sg, xi):
    z = (np.asarray(x, float) - mu) / sg
    if abs(xi) < 1e-9:
        return np.exp(-np.exp(-z))
    t = 1 + xi * z
    return np.where(t > 0, np.exp(-np.power(np.maximum(t, 1e-300), -1 / xi)), 0.0 if xi > 0 else 1.0)


def gev_logpdf(x, mu, sg, xi):
    z = (np.asarray(x, float) - mu) / sg
    if abs(xi) < 1e-9:
        return -math.log(sg) - z - np.exp(-z)
    t = 1 + xi * z
    if np.any(t <= 0):
        return np.full_like(z, -np.inf)
    return -math.log(sg) - (1 + 1 / xi) * np.log(t) - np.power(t, -1 / xi)


def fit_all(rho, n, xs, cdf):
    g = 1 - rho
    xs, cdf = np.asarray(xs), np.asarray(cdf)
    sel = (cdf >= 0.01) & (cdf <= 0.999)
    xk, ck = xs[sel], cdf[sel]
    med = float(np.interp(0.5, cdf, xs))
    best = None
    for xi0 in (0.1, -0.1, 0.0):
        r = optimize.minimize(lambda q: float(np.max(np.abs(ck - gev_cdf(xk, q[0], math.exp(q[1]), q[2])))), [med - 0.3 / g, math.log(1 / g), xi0], method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 6000})
        if best is None or r.fun < best.fun:
            best = r
    r0 = optimize.minimize(lambda q: float(np.max(np.abs(ck - gev_cdf(xk, q[0], math.exp(q[1]), 0.0)))), [med - 0.3 / g, math.log(1 / g)], method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 6000})
    # Populations-ML: Zellenmasse aus F (Zelle (x_{i-1}, x_i]), Vertreter Zellenmitte
    full = cdf >= 1e-7
    xf, cf = xs[full], cdf[full]
    mass = np.diff(np.concatenate([[0.0], cf]))
    mid = xf - STEP / 2
    keep = mass > 0
    mass, mid = mass[keep] / mass[keep].sum(), mid[keep]

    def nlog(q, fix0=False):
        v = gev_logpdf(mid, q[0], math.exp(q[1]), 0.0 if fix0 else q[2])
        return float(-np.sum(mass * v)) if np.all(np.isfinite(v)) else 1e12

    bm = None
    for xi0 in (0.1, -0.05):
        r = optimize.minimize(nlog, [best.x[0], best.x[1], xi0], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 6000})
        if bm is None or r.fun < bm.fun:
            bm = r
    bm0 = optimize.minimize(lambda q: nlog(q, True), [best.x[0], best.x[1]], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 6000})
    return {
        "x": [float(v) for v in xk], "cdf": [round(float(v), 8) for v in ck],
        "ks_gev": {"mu": float(best.x[0]), "sigma": float(math.exp(best.x[1])), "xi": float(best.x[2]), "ks": float(best.fun)},
        "ks_gumbel": {"mu": float(r0.x[0]), "sigma": float(math.exp(r0.x[1])), "ks": float(r0.fun)},
        "ml_gev": {"mu": float(bm.x[0]), "sigma": float(math.exp(bm.x[1])), "xi": float(bm.x[2])},
        "ml_gumbel": {"mu": float(bm0.x[0]), "sigma": float(math.exp(bm0.x[1]))},
        "sigma_grenzwert": 1.0 / g,
    }


def raster(rho, n):
    lo = max(STEP, math.floor((K.gumbel_day_quantile(rho, n, 1e-3) - 4.0) / STEP) * STEP)
    hi = K.gumbel_day_quantile(rho, n, 1 - 1e-5, theta=1.0)
    return np.arange(lo, math.ceil(hi / STEP) * STEP + STEP, STEP)


def build(workers, rhos=RHOS, ns=NS):
    t0 = time.time()
    tasks = [(rho, n, float(x)) for rho in rhos for n in ns for x in raster(rho, n)]
    tasks.sort(key=lambda t: -t[2])
    with Pool(workers) as pool:
        res = pool.map(task, tasks, chunksize=4)
    by = {}
    for rho, n, x, c in res:
        by.setdefault((rho, n), []).append((x, c))
    out = {}
    for (rho, n), rows in sorted(by.items()):
        rows.sort()
        xs = [r[0] for r in rows]
        cdf = np.maximum.accumulate([r[1] for r in rows])
        out.setdefault(str(rho), {})[str(n)] = fit_all(rho, n, xs, cdf)
    return {
        "quelle": "tools/clu_make_blocks.py (Gitterkern clu_kernel.py, exakte Verteilung des Blockmaximums; Anpassung mit scipy im Werkzeug); eigene Rechnung, kein Zufall",
        "stand": STAND, "gitterweite_h": H, "raster_x": STEP, "n_liste": list(NS),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "rechenzeit_s": round(time.time() - t0, 1), "daten": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.write:
        data = build(a.workers)
        OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"geschrieben: {OUT.name}, {OUT.stat().st_size / 1024:.0f} KB, {data['rechenzeit_s']} s")
    elif a.check:
        old = json.loads(OUT.read_text(encoding="utf-8"))["daten"]["0.5"]["64"]
        new = build(a.workers, rhos=(0.5,), ns=(64,))["daten"]["0.5"]["64"]
        bad = []
        for key in ("ks_gev", "ml_gev"):
            for f in ("xi", "sigma"):
                if abs(old[key][f] - new[key][f]) > 1e-6 * max(1.0, abs(old[key][f])):
                    bad.append(f"{key}.{f}: {old[key][f]} gegen {new[key][f]}")
        if max(abs(a_ - b_) for a_, b_ in zip(old["cdf"], new["cdf"])) > 1e-8:
            bad.append("cdf weicht ab")
        print("Abweichungen:", len(bad), *bad)
        sys.exit(1 if bad else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
