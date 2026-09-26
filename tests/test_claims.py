"""Jede Zahl aus README und App über die echten Auswertungsfunktionen (Iterationen mit Band, Grenzen mit Sicherheitsabstand: Rundung kann plattformabhängig kleine Unterschiede machen)."""

import statistics

import pytest

import ipm_constants as C
import ipm_evaluation as ev
import ipm_ipm as P
import ipm_scenario as S
from ipm_evaluation import Settings


def near(x, want, rel=0.03, tol=2):
    return abs(x - want) <= max(rel * abs(want), tol)


def test_readme_iterations_over_n_for_every_method():
    rows = {r["n"]: r for r in ev.size_sweep(Settings("random"))}
    want = {"mehrotra": (4, 5, 6, 8, 8, 9, 9), "long": (11, 12, 14, 17, 19, 21, 21), "affine": (10, 11, 13, 16, 19, 19, 19), "short": (79, 126, 191, 274, 343, 410, 448)}
    for m, vals in want.items():
        assert all(near(rows[n]["iterations"][m], v, tol=2 if m != "short" else 4, rel=0.03) for n, v in zip(C.SWEEP_SIZES, vals)), m
    assert all(rows[n]["optimal"][m] == 5 for n in rows for m in P.METHODS)
    assert rows[40]["iterations"]["mehrotra"] < 2.6 * rows[2]["iterations"]["mehrotra"] and rows[40]["iterations"]["short"] > 5 * rows[2]["iterations"]["short"]


def test_readme_operations_against_the_simplex():
    rows = {r["n"]: r for r in ev.size_sweep(Settings("random"))}
    assert [round(rows[n]["ratio"], 1) for n in C.SWEEP_SIZES] == pytest.approx([20.5, 18.8, 17.9, 18.4, 15.7, 24.4, 20.0], abs=2.0)
    assert [rows[n]["pivots"] for n in (2, 4, 8, 16, 24, 32, 40)] == pytest.approx([1, 2, 4, 9, 15, 14, 21], abs=2)


def test_readme_accuracy_and_floor():
    rows = {r["k"]: r for r in ev.eps_sweep(Settings("random", 16, 16, 35))}
    assert [rows[k]["iterations"] for k in (2, 4, 6, 8, 10, 12, 13)] == pytest.approx([5, 7, 7, 8, 9, 9, 10], abs=1) and all(rows[k]["optimal"] == 5 for k in (2, 4, 6, 8, 10, 12, 13))
    assert 0 <= rows[14]["optimal"] <= 5                                                            # 10^-14: an der Rundungsgrenze, je nach Instanz und Plattform (Windows: 4 von 5)
    for k, want in ((8, "optimal"), (12, "optimal")):
        for sd in C.SWEEP_SEEDS:
            assert P.ipm(S.generate("random", 16, 16, C.DENSITY, sd), "mehrotra", eps=10.0 ** -k).status == want


def test_readme_condition_numbers():
    top = [max(P.ipm(S.generate("random", 16, 16, C.DENSITY, sd), "mehrotra", eps=1e-8, keep_cond=True).cond) for sd in C.SWEEP_SEEDS]
    assert max(top) < 1e8 and min(top) > 1e3
    a = ev.analyse(Settings("random", 16, 16, 35, eps_i=6))
    assert max(a.res.cond) > 1e12


def test_readme_short_step_against_theory():
    rows = {r["n"]: r for r in ev.theory_compare(Settings("random"))}
    assert [round(100 * rows[n]["share"]) for n in (4, 8, 16, 32)] == pytest.approx([84, 85, 87, 89], abs=2)
    assert [round(rows[n]["theory"]) for n in (4, 8, 16, 32)] == pytest.approx([148, 225, 315, 463], rel=0.03)


def test_readme_klee_minty_cube():
    sw = ev.cube_sweep(Settings())
    r = {row["n"]: row for row in sw["rows"]}
    assert [r[n]["iterations"] for n in (2, 6, 8, 10, 12, 14)] == pytest.approx([5, 9, 12, 13, 16, 17], abs=1) and r[14]["pivots"] == 16383
    assert [round(r[n]["ratio"], 2) for n in (2, 4, 6, 7, 8, 10, 14)] == pytest.approx([8.56, 3.01, 1.39, 0.85, 0.56, 0.18, 0.019], rel=0.12, abs=0.005) and sw["crossover"] == 7
    long = ev.cube_sweep(Settings(method="long"))
    assert long["crossover"] in (7, 8, 9, 10) and all(row["status"] == "optimal" for row in long["rows"])


def test_readme_start_point_and_scaling():
    rows = {r["factor"]: r for r in ev.start_sweep(Settings("random", 16, 16, 35))}
    assert [rows[f]["mehrotra"]["optimal"] for f in C.START_FACTORS] == [0, 5, 5, 5, 5, 5, 5] and [rows[f]["long"]["optimal"] for f in C.START_FACTORS][:3] == [0, 3, 5]
    assert [rows[f]["mehrotra"]["iterations"] for f in (0.01, 1, 100, 1e8)] == pytest.approx([25, 8, 13, 16], abs=4)
    sc = {r["k"]: r for r in ev.scale_sweep(Settings("random"))}
    assert [sc[k]["mehrotra"]["right"] for k in (0, 4, 8, 12, 14, 16)] == [5, 5, 5, 4, 3, 3] and all(sc[k]["short"]["right"] == 5 for k in sc)
    assert [sc[k]["long"]["right"] for k in (0, 2, 4, 6, 8)] == [5, 5, 3, 1, 0] and sc[16]["affine"]["right"] == 0 and sc[16]["long"]["right"] == 0
    assert all(sc[k][m]["wrong"] == 0 for k in sc for m in P.METHODS)
    assert [sc[k]["mehrotra"]["iterations"] for k in (0, 2, 4, 8)] == pytest.approx([6, 11, 17, 24], abs=4)


def test_readme_hundred_instances_all_methods_optimal():
    for kind, m, n, med_want, max_want in (("random", 12, 12, {"mehrotra": 7, "long": 16, "affine": 15, "short": 236}, 300), ("mixed", 10, 10, {"mehrotra": 7, "long": 15, "affine": 14, "short": 192}, 300)):
        for method in P.METHODS:
            its, ok = [], 0
            for sd in range(100):
                r = P.ipm(S.generate(kind, m, n, C.DENSITY, sd), method, eps=1e-8)
                its.append(r.iterations)
                ok += r.status == "optimal"
            assert ok == 100 and near(statistics.median(its), med_want[method], rel=0.05, tol=2) and max(its) < max_want, (kind, method)


def test_readme_affine_scaling_never_failed_on_the_random_instances():
    bad = 0
    for sd in range(200):
        inst = S.generate("random", 8, 8, C.DENSITY, sd)
        a, m = P.ipm(inst, "affine", eps=1e-8), P.ipm(inst, "mehrotra", eps=1e-8)
        bad += a.status != "optimal" or a.iterations > 3 * m.iterations
    assert bad == 0


def test_readme_certificates_and_named_instances():
    for kind, status in (("infeasible", "infeasible"), ("unbounded", "unbounded")):
        for method in ("mehrotra", "short", "affine"):
            assert ev.analyse(Settings(kind, method=method)).res.status == status
    named = {kind: ev.analyse(Settings(kind)).res.iterations for kind in ("textbook", "centre", "degenerate")}
    assert near(named["textbook"], 4, tol=1) and near(named["centre"], 7, tol=1) and near(named["degenerate"], 4, tol=1)
