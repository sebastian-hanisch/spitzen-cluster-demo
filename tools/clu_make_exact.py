"""Erzeugt und prüft `clu_exact.json`: die exakte Verteilung des Tagesmaximums (Experiment 2) und θ(u) aus dem Kern (Experiment 1). Kein Zufall.

Tagesmaximum = größte Wartezeit unter n aufeinanderfolgenden Kunden der stationär gestarteten M/M/1-Schlange (Dienstrate 1, Last ρ); P(max ≤ x) = π Q^(n−1) 1 auf dem Gitter h = 0,1.
Gespeichert: die Verteilungsfunktion auf dem Raster der Breite 0,5 (nur dort, wo sie zwischen 1e-6 und 1 − 1e-6 liegt) für n = 16 bis 1024, und für die Quantile 0,1 / 0,5 / 0,9 / 0,99 / 0,999
die Werte, die auf dem Feinraster 0,1 um die grobe Schätzung berechnet und linear interpoliert sind. Dazu θ(u) = (1 − λ_u)/F̄_h(u) bei den Randanteilen F̄ = 1e-2 bis 1e-6 mit zwei Gitterweiten.

Aufruf (Test-venv, im Ordner der Demo; OMP_NUM_THREADS=1):
    python tools/clu_make_exact.py --write [--workers 6]
    python tools/clu_make_exact.py --check       # rechnet Last 0,5 vollständig neu und vergleicht mit relativer Toleranz 1e-9"""

import argparse
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import clu_kernel as K  # noqa: E402
import clu_queue as Q  # noqa: E402

OUT = ROOT / "clu_exact.json"
STAND = "2026-11-20"
RHOS = (0.5, 0.8, 0.9)
NS = (16, 32, 64, 128, 256, 512, 1024)
QS = (0.1, 0.5, 0.9, 0.99, 0.999)
H = 0.1
STEP = 0.5
FBARS = (1e-2, 1e-3, 1e-4, 1e-6)
THETA_H = (0.1, 0.2)
EDGE = 1e-6

_GRIDS: dict = {}


def _grid(rho: float) -> K.Grid:
    if rho not in _GRIDS:
        hi = K.gumbel_day_quantile(rho, max(NS), 1 - EDGE, theta=1.0) + 12.0
        _GRIDS[rho] = K.Grid(rho, H, hi)
    return _GRIDS[rho]


def x_raster(rho: float) -> np.ndarray:
    hi = K.gumbel_day_quantile(rho, max(NS), 1 - EDGE, theta=1.0)
    return np.arange(STEP, np.ceil(hi / STEP) * STEP + STEP, STEP)


def task_cdf(args):
    rho, xs = args
    g = _grid(rho)
    return [(float(x), g.day_max_cdf_at(float(x), NS)) for x in xs]


def task_quantile(args):
    rho, n, q, x0 = args
    g = _grid(rho)
    xf = np.unique(np.round(np.arange(x0 - 0.8, x0 + 0.85, 0.1) / H) * H)
    xf = xf[xf > 0]
    cf = np.array([g.day_max_cdf(float(x), n) for x in xf])
    return rho, n, q, (float(np.interp(q, cf, xf)) if cf[0] < q < cf[-1] else None)


def task_theta(args):
    rho, f_bar, h = args
    g = K.grid_for(rho, h, min(FBARS))
    u = Q.mm1_threshold(rho, f_bar)
    j = g.index(u)
    return rho, f_bar, h, {"u": j * h, "theta": g.theta_at(u), "f_bar_gitter": float(g.surv[j])}


def build(workers: int, rhos=RHOS, with_quantiles=True) -> dict:
    t0 = time.time()
    day = {}
    with Pool(workers) as pool:
        for rho in rhos:
            xs = x_raster(rho)
            chunks = [(rho, xs[i::workers * 4]) for i in range(workers * 4)]
            rows = sorted((r for part in pool.map(task_cdf, chunks) for r in part))
            xv = np.array([r[0] for r in rows])
            cdf = np.array([r[1] for r in rows])          # (Zahl x, Zahl n)
            per_n = {}
            for i, n in enumerate(NS):
                c = cdf[:, i]
                keep = (c >= EDGE) & (c <= 1 - EDGE)
                lo, hi = int(np.argmax(keep)), int(len(keep) - np.argmax(keep[::-1]))
                per_n[str(n)] = {"x0": float(xv[lo]), "cdf": [round(float(v), 8) for v in c[lo:hi]]}
            quant = {}
            if with_quantiles:
                tasks = []
                for n in NS:
                    d = per_n[str(n)]
                    xs_n = d["x0"] + STEP * np.arange(len(d["cdf"]))
                    for q in QS:
                        tasks.append((rho, n, q, float(np.interp(q, d["cdf"], xs_n))))
                for _rho, n, q, val in pool.map(task_quantile, tasks):
                    quant.setdefault(str(n), {})[str(q)] = None if val is None else round(val, 4)
            day[str(rho)] = {"gumbel_hi": float(xv[-1]), "n": per_n, "quantile": quant}
        theta_tasks = [(rho, fb, h) for rho in rhos for fb in FBARS for h in THETA_H]
        theta = {}
        for rho, fb, h, val in pool.map(task_theta, theta_tasks):
            theta.setdefault(str(rho), {}).setdefault(str(fb), {})[str(h)] = {k: float(v) for k, v in val.items()}
    return {
        "quelle": "tools/clu_make_exact.py (Gitterkern der Lindley-Kette, clu_kernel.py); eigene Rechnung, kein Zufall",
        "stand": STAND, "gitterweite_h": H, "raster_x": STEP, "n_liste": list(NS), "quantile": list(QS), "raender": list(FBARS),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "rechenzeit_s": round(time.time() - t0, 1),
        "tag": day, "theta_u": theta,
    }


def compare(a, b, path=""):
    """Rekursiver Vergleich mit relativer Toleranz 1e-9 für Zahlen; gibt die Liste der Abweichungen zurück."""
    bad = []
    if isinstance(a, dict):
        for k in a:
            if k in ("rechenzeit_s", "software", "stand"):
                continue
            if k not in b:
                bad.append(f"{path}/{k} fehlt")
            else:
                bad += compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            bad.append(f"{path}: Länge {len(a)} gegen {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                bad += compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > 1e-9 * max(1.0, abs(a)) and not (abs(a) < 1e-7 and abs(a - b) < 1e-8):
            bad.append(f"{path}: {a} gegen {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} gegen {b!r}")
    return bad


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
        old = json.loads(OUT.read_text(encoding="utf-8"))
        new = build(a.workers, rhos=(0.5,))
        sub = {"tag": {"0.5": old["tag"]["0.5"]}, "theta_u": {"0.5": old["theta_u"]["0.5"]}}
        mine = {"tag": {"0.5": new["tag"]["0.5"]}, "theta_u": {"0.5": new["theta_u"]["0.5"]}}
        bad = compare(sub, mine)
        print("Abweichungen:", len(bad))
        for b in bad[:10]:
            print("  ", b)
        sys.exit(1 if bad else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
