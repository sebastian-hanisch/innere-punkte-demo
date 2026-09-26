"""Auswertung der Innere-Punkte-Verfahren: ein Lauf gegen den Simplex, Iterationen über n / Genauigkeit / Startpunkt, Kurzschritt gegen die Theorie, Operationen, Klee-Minty-Würfel, Skalierung, Zentralpfad."""

import math
import statistics
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ipm_algorithm as A
import ipm_constants as C
import ipm_ipm as P
import ipm_scenario as S


@dataclass(frozen=True)
class Settings:
    kind: str = "centre"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    method: str = "mehrotra"
    eps_i: int = C.DEFAULT_EPS_I
    start_i: int = C.DEFAULT_START_I
    scale_i: int = C.DEFAULT_SCALE_I

    @property
    def eps(self):
        return 10.0 ** -C.EPS_EXPS[min(max(self.eps_i, 0), len(C.EPS_EXPS) - 1)]

    @property
    def start_factor(self):
        return C.START_FACTORS[min(max(self.start_i, 0), len(C.START_FACTORS) - 1)]

    @property
    def scale_exp(self):
        return C.SCALE_EXPS[min(max(self.scale_i, 0), len(C.SCALE_EXPS) - 1)]


def base_instance(s):
    return S.generate(s.kind, s.m, s.n, C.DENSITY, s.seed)


def instance_of(s):
    return S.column_scaled(base_instance(s), s.scale_exp)


def simplex_flops(inst, solution):
    """Operationsmodell des dichten Tableau-Simplex: je Pivot 2 (m+1)(Spalten+1)."""
    return solution.pivots * 2 * (inst.m + 1) * (A.standard_form(inst)[2]["ncols"] + 1)


@dataclass
class Analysis:
    inst: object
    base: object
    res: object                      # IPMResult
    simplex: object
    ref_status: str
    ref_obj: float                   # Optimalwert des Simplex auf der unskalierten Instanz
    obj_error: float
    flops_ipm: int
    flops_simplex: int
    method: str = "mehrotra"


@lru_cache(maxsize=64)
def analyse(s):
    inst, base = instance_of(s), base_instance(s)
    res = P.ipm(inst, s.method, eps=s.eps, keep=True, keep_cond=True, start_factor=s.start_factor)
    sol, ref = A.solve(inst), A.solve(base)
    ref_obj = ref.obj if ref.status == "optimal" else float("nan")
    return Analysis(inst, base, res, sol, ref.status, ref_obj, (res.obj - ref_obj) if res.x and ref.status == "optimal" else float("nan"), res.flops,
                    simplex_flops(inst, sol) if sol.status == "optimal" else 0, s.method)


def _random(n, seed, m=None):
    return S.generate("random", n if m is None else m, n, C.DENSITY, seed)


def _med(values):
    return statistics.median(values) if values else float("nan")


def _run(inst, method, eps, **kw):
    return P.ipm(inst, method, eps=eps, **kw)


@lru_cache(maxsize=32)
def size_sweep(s):
    """Zufallsinstanzen m = n über die Größen: Iterationen aller vier Verfahren (Median, Spanne), Operationen des gewählten Verfahrens gegen den Simplex."""
    rows = []
    for n in C.SWEEP_SIZES:
        per = {m: [] for m in P.METHODS}
        ok = {m: 0 for m in P.METHODS}
        fl, fs, pv = [], [], []
        for sd in C.SWEEP_SEEDS:
            inst = _random(n, sd)
            sol = A.solve(inst)
            pv.append(sol.pivots), fs.append(simplex_flops(inst, sol))
            for m in P.METHODS:
                r = _run(inst, m, s.eps, start_factor=s.start_factor)
                per[m].append(r.iterations)
                ok[m] += r.status == "optimal"
                if m == s.method:
                    fl.append(r.flops)
        rows.append({"n": n, "iterations": {m: _med(per[m]) for m in P.METHODS}, "min": {m: min(per[m]) for m in P.METHODS}, "max": {m: max(per[m]) for m in P.METHODS},
                     "optimal": ok, "flops_ipm": _med(fl), "flops_simplex": _med(fs), "pivots": _med(pv), "ratio": _med(fl) / max(_med(fs), 1.0)})
    return rows


@lru_cache(maxsize=32)
def eps_sweep(s):
    """Iterationen über die Genauigkeit (Zufall m = n aus dem Regler, gewähltes Verfahren): Median und Anteil erfolgreicher Läufe; unterhalb der Rundungsgrenze scheitert das Verfahren."""
    rows = []
    for k in C.EPS_SWEEP_EXPS:
        its, ok = [], 0
        for sd in C.SWEEP_SEEDS:
            r = _run(_random(s.n, sd), s.method, 10.0 ** -k, start_factor=s.start_factor)
            its.append(r.iterations)
            ok += r.status == "optimal"
        rows.append({"k": k, "iterations": _med(its), "optimal": ok})
    return rows


@lru_cache(maxsize=32)
def start_sweep(s):
    """Iterationen über den Faktor auf den Startpunkt (Zufall m = n aus dem Regler): Mehrotra und Langschritt; zu kleine Startpunkte lassen das Verfahren stillstehen."""
    rows = []
    for f in C.START_FACTORS:
        row = {"factor": f}
        for m in ("mehrotra", "long"):
            its, ok = [], 0
            for sd in C.SWEEP_SEEDS:
                r = _run(_random(s.n, sd), m, s.eps, start_factor=f)
                its.append(r.iterations)
                ok += r.status == "optimal"
            row[m] = {"iterations": _med(its), "optimal": ok}
        rows.append(row)
    return rows


@lru_cache(maxsize=32)
def theory_compare(s):
    """Kurzschritt: gemessene Iterationen gegen die Schranke ln(mu0 / eps) / -ln(1 - 0.4 / sqrt(N)) (Wright 1997), Zufall m = n."""
    rows = []
    for n in C.THEORY_SIZES:
        meas, theo = [], []
        for sd in C.SWEEP_SEEDS:
            r = _run(_random(n, sd), "short", s.eps)
            meas.append(r.iterations)
            theo.append(math.log(r.mu_path[0] / s.eps) / -math.log(1.0 - 0.4 / math.sqrt(r.N)))
        rows.append({"n": n, "iterations": _med(meas), "theory": _med(theo), "share": _med([a / b for a, b in zip(meas, theo)])})
    return rows


@lru_cache(maxsize=32)
def cube_sweep(s):
    """Klee-Minty-Würfel n = 2..14: Iterationen des gewählten Verfahrens gegen die 2^n - 1 Pivots des Simplex, Operationen im Modell, Kreuzungspunkt (kleinstes n, ab dem das Verfahren dauerhaft weniger braucht)."""
    rows = []
    for n in C.CUBE_SIZES:
        inst = S.klee_minty_instance(n)
        r = _run(inst, s.method, s.eps, start_factor=s.start_factor)
        sol = A.solve(inst)
        fs = simplex_flops(inst, sol)
        rows.append({"n": n, "iterations": r.iterations, "pivots": sol.pivots, "flops_ipm": r.flops, "flops_simplex": fs, "ratio": r.flops / max(fs, 1), "status": r.status})
    cross = None
    for r in rows:
        if all(q["flops_ipm"] < q["flops_simplex"] for q in rows if q["n"] >= r["n"]):
            cross = r["n"]
            break
    return {"rows": rows, "crossover": cross}


@lru_cache(maxsize=32)
def scale_sweep(s):
    """Schlechte Skalierung: dieselben fünf Zufallsinstanzen (8 × 8) mit Spalten über 10^k gestreut, alle vier Verfahren: optimal richtig / optimal falsch / Stillstand oder Verdacht, Iterationen (Median der Optimalen)."""
    rows = []
    for k in C.SCALE_EXPS:
        row = {"k": k}
        for m in P.METHODS:
            right = wrong = other = 0
            its = []
            for sd in C.SWEEP_SEEDS:
                base = _random(8, sd)
                true = A.solve(base).obj
                r = _run(S.column_scaled(base, k), m, 1e-8)
                if r.status == "optimal":
                    if abs(r.obj - true) / (1.0 + abs(true)) > 1e-3:
                        wrong += 1
                    else:
                        right += 1
                        its.append(r.iterations)
                else:
                    other += 1
            row[m] = {"right": right, "wrong": wrong, "other": other, "iterations": _med(its)}
        rows.append(row)
    return rows


@lru_cache(maxsize=32)
def method_table(s):
    """Alle vier Verfahren auf der gewählten Instanz."""
    inst = instance_of(s)
    out = []
    for m in P.METHODS:
        r = _run(inst, m, s.eps, start_factor=s.start_factor)
        out.append({"method": m, "status": r.status, "iterations": r.iterations, "obj": r.obj, "flops": r.flops, "final_mu": r.mu_path[-1] if r.mu_path else float("nan")})
    return out


@lru_cache(maxsize=32)
def central_path(s):
    """Zentralpfad einer Instanz mit zwei Variablen: Strukturvariablen der Punkte z(mu) für mu von der Größe des Startpunkts bis 1e-6 davon (geometrisch); None, wenn die Instanz keinen Pfad hat."""
    inst = instance_of(s)
    if inst.n != 2:
        return None
    M, b, c, n, _ = P.standard_form(inst)
    if M is None:
        return None
    z0, y0, s0 = P.mehrotra_start(M, b, c)
    mu0 = float(z0 @ s0) / len(z0)
    pts, cur = [], (z0, y0, s0)
    for mu in mu0 * np.geomspace(1.0, 1e-6, C.PATH_POINTS):
        try:
            cur = P.central_point(M, b, c, float(mu), start=cur)
        except np.linalg.LinAlgError:
            break
        pts.append(cur[0][:n].copy())
    return np.array(pts) if pts else None
