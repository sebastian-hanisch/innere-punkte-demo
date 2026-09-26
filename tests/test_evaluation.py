"""Auswertung: Einzellauf gegen den Simplex, Größen-/Genauigkeits-/Startpunkt-Kurven, Kurzschritt gegen Theorie, Klee-Minty-Würfel, Skalierung, Zentralpfad."""

import math

import numpy as np
import pytest

import ipm_constants as C
import ipm_evaluation as ev
import ipm_ipm as P
import ipm_scenario as S
from ipm_evaluation import Settings


def test_settings_clamp_the_index_controls():
    s = Settings(eps_i=99, start_i=-3, scale_i=99)
    assert s.eps == 1e-14 and s.start_factor == 0.001 and s.scale_exp == 16
    assert Settings().eps == 1e-8 and Settings().start_factor == 1 and Settings(start_i=6).start_factor == 1e8


def test_analyse_compares_with_the_simplex_and_is_cached():
    a = ev.analyse(Settings())
    assert a is ev.analyse(Settings())
    assert a.res.status == "optimal" and a.simplex.status == "optimal" and a.ref_obj == pytest.approx(720.0) and abs(a.obj_error) < 1e-5
    assert a.flops_simplex == 400 and a.flops_ipm == a.res.iterations * P.flops_per_iteration(a.res.m, a.res.N, 2) and a.method == "mehrotra"
    assert len(a.res.points) == a.res.iterations + 1 and len(a.res.cond) == a.res.iterations
    inf, unb = ev.analyse(Settings("infeasible")), ev.analyse(Settings("unbounded"))
    assert inf.res.status == "infeasible" and inf.ref_status == "infeasible" and math.isnan(inf.obj_error) and unb.res.status == "unbounded" and unb.flops_simplex == 0


def test_scaled_instance_keeps_the_true_reference_value():
    a = ev.analyse(Settings("random", 6, 8, 35, scale_i=3))
    assert a.inst != a.base and a.ref_obj == pytest.approx(ev.analyse(Settings("random", 6, 8, 35)).ref_obj)
    assert S.column_scaled(a.base, 0) is a.base


def test_size_sweep_iterations_grow_slowly_and_the_simplex_stays_ahead():
    rows = ev.size_sweep(Settings("random"))
    assert [r["n"] for r in rows] == list(C.SWEEP_SIZES)
    for m in P.METHODS:
        assert all(r["optimal"][m] == 5 for r in rows)
    mehrotra = [r["iterations"]["mehrotra"] for r in rows]
    assert mehrotra[0] <= 6 and mehrotra[-1] <= 12 and mehrotra[-1] < 3 * mehrotra[0] and all(b >= a - 1 for a, b in zip(mehrotra, mehrotra[1:]))
    short = [r["iterations"]["short"] for r in rows]
    assert short == sorted(short) and short[-1] > 4 * short[0]
    assert all(r["iterations"]["mehrotra"] < r["iterations"]["long"] for r in rows) and all(10 < r["ratio"] < 40 for r in rows) and ev.size_sweep(Settings("random")) is rows


def test_eps_sweep_grows_by_about_one_iteration_per_digit_and_fails_near_machine_precision():
    rows = ev.eps_sweep(Settings("random", 16, 16, 35))
    its = {r["k"]: r["iterations"] for r in rows}
    assert [r["k"] for r in rows] == list(C.EPS_SWEEP_EXPS) and its[12] - its[2] <= 6 and its[13] >= its[2]
    assert all(r["optimal"] == 5 for r in rows if r["k"] <= 12)


def test_start_sweep_shows_the_stall_for_tiny_starts_and_mild_cost_for_large_ones():
    rows = {r["factor"]: r for r in ev.start_sweep(Settings("random", 16, 16, 35))}
    assert rows[0.001]["mehrotra"]["optimal"] == 0 and rows[1]["mehrotra"]["optimal"] == 5 and rows[1e8]["mehrotra"]["optimal"] == 5
    assert rows[1e8]["mehrotra"]["iterations"] < 3 * rows[1]["mehrotra"]["iterations"] and rows[0.01]["long"]["iterations"] > 4 * rows[1]["long"]["iterations"]


def test_theory_compare_short_step_stays_below_the_bound():
    rows = ev.theory_compare(Settings("random"))
    assert [r["n"] for r in rows] == list(C.THEORY_SIZES) and all(0.75 < r["share"] < 1.0 for r in rows)


def test_cube_sweep_crossover_and_pivots():
    sw = ev.cube_sweep(Settings())
    rows = sw["rows"]
    assert [r["n"] for r in rows] == list(C.CUBE_SIZES) and all(r["pivots"] == 2 ** r["n"] - 1 and r["status"] == "optimal" for r in rows)
    assert sw["crossover"] in (6, 7, 8) and rows[-1]["iterations"] <= 25 and rows[-1]["ratio"] < 0.05 and rows[0]["ratio"] > 4
    ratios = [r["ratio"] for r in rows]
    assert ratios == sorted(ratios, reverse=True)


def test_scale_sweep_mehrotra_and_short_step_are_robust_long_step_and_affine_are_not():
    rows = {r["k"]: r for r in ev.scale_sweep(Settings("random"))}
    assert all(rows[0][m]["right"] == 5 for m in P.METHODS) and all(r[m]["wrong"] == 0 for r in rows.values() for m in P.METHODS)
    assert all(rows[k]["short"]["right"] == 5 for k in rows) and all(rows[k]["mehrotra"]["right"] >= 3 for k in rows)
    assert rows[8]["long"]["right"] == 0 and rows[12]["affine"]["right"] == 0 and rows[12]["mehrotra"]["iterations"] > 2 * rows[0]["mehrotra"]["iterations"]


def test_method_table_lists_all_methods_in_order():
    tab = ev.method_table(Settings("random", 16, 16, 35))
    assert [t["method"] for t in tab] == list(P.METHODS) and all(t["status"] == "optimal" for t in tab)
    its = {t["method"]: t["iterations"] for t in tab}
    assert its["mehrotra"] < its["affine"] <= its["long"] + 2 < its["short"] and len({round(t["obj"], 4) for t in tab}) == 1


def test_central_path_lies_on_the_path_and_ends_at_the_optimum():
    s = Settings("textbook")
    path = ev.central_path(s)
    assert path.shape == (C.PATH_POINTS, 2) and np.allclose(path[-1], [2.0, 6.0], atol=1e-3)
    M, b, c, n, _ = P.standard_form(S.textbook_instance())
    z0, y0, s0 = P.mehrotra_start(M, b, c)
    mu = float(z0 @ s0) / len(z0)
    z, y, sl = P.central_point(M, b, c, mu)
    assert np.allclose(z * sl, mu, rtol=1e-6) and np.allclose(M @ z, b, atol=1e-8) and np.allclose(M.T @ y + sl, c, atol=1e-8)
    assert ev.central_path(Settings("centre")) is None
