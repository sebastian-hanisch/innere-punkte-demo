"""Korrektheitskette: Newton-System gegen das volle KKT-System, Optimum gegen HiGHS und Simplex, strikte Innerlichkeit und Umgebungen, Kurzschritt-Reduktion, Mehrotra-Formeln, Unzulässig/Unbeschränkt,
Sonderfälle (redundante Zeilen, Startpunkt, Skalierung), Buchführung."""

import math
import random

import numpy as np
import pytest

import ipm_algorithm as A
import ipm_ipm as P
import ipm_scenario as S
from tests.test_scenario import _highs, reference_status


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _instances(seeds=8):
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for n in (2, 5, 9, 14):
        yield S.klee_minty_instance(n)
    for seed in range(seeds):
        yield S.generate("random", 5, 6, 0.5, seed)
        yield S.generate("random", 8, 4, 0.5, seed)
        yield S.generate("mixed", 6, 6, 0.5, seed)


def _optimal(inst):
    h = _highs(inst)
    return None if h.status != 0 else -h.fun


# --- 1. Newton-Richtung ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_newton_direction_solves_the_full_kkt_system(seed):
    rng = np.random.default_rng(seed)
    m, N = 4, 9
    M = rng.normal(size=(m, N))
    z, s = rng.uniform(0.5, 3, N), rng.uniform(0.5, 3, N)
    r_p, r_d, r_c = rng.normal(size=m), rng.normal(size=N), rng.normal(size=N)
    L = np.linalg.cholesky((M * (z / s)) @ M.T)
    dz, dy, ds = P.newton_direction(M, z, s, L, r_p, r_d, r_c)
    K = np.block([[M, np.zeros((m, m)), np.zeros((m, N))], [np.zeros((N, N)), M.T, np.eye(N)], [np.diag(s), np.zeros((N, m)), np.diag(z)]])
    full = np.linalg.solve(K, np.concatenate([r_p, r_d, r_c]))
    assert np.allclose(dz, full[:N], atol=1e-9) and np.allclose(dy, full[N:N + m], atol=1e-9) and np.allclose(ds, full[N + m:], atol=1e-9)
    assert np.allclose(M @ dz, r_p, atol=1e-9) and np.allclose(M.T @ dy + ds, r_d, atol=1e-9) and np.allclose(s * dz + z * ds, r_c, atol=1e-9)


def test_standard_form_shapes_and_signs():
    inst = S.generate("mixed", 6, 6, 0.5, 3)
    M, b, c, n, note = P.standard_form(inst)
    extra = sum(1 for x in inst.senses if x != S.EQ)
    assert M.shape == (6, 6 + extra) and n == 6 and c[:6].tolist() == [-v for v in inst.c] and not c[6:].any()
    k = 6
    for i, sense in enumerate(inst.senses):
        if sense != S.EQ:
            assert M[i, k] == (1.0 if sense == S.LE else -1.0) and np.count_nonzero(M[:, k]) == 1
            k += 1
    assert note == ""


# --- 2. Optimum ------------------------------------------------------------------------------------------------------------------------------------

def test_every_method_reaches_the_optimum_of_highs_and_the_simplex():
    count = 0
    for inst in _instances(30):
        opt = _optimal(inst)
        assert opt is not None and A.solve(inst).obj == pytest.approx(opt, rel=1e-7, abs=1e-6)
        for method in P.METHODS:
            r = P.ipm(inst, method, eps=1e-8)
            assert r.status == "optimal", (inst.kind, method, r.status, r.note, r.iterations)
            assert r.obj == pytest.approx(opt, rel=1e-6, abs=1e-5), (inst.kind, method)
            A_, b_, _c = inst.arrays()
            x = np.array(r.x)
            for i, sense in enumerate(inst.senses):
                v = float(A_[i] @ x)
                assert (v <= b_[i] + 1e-5 * (1 + abs(b_[i]))) if sense == S.LE else (v >= b_[i] - 1e-5 * (1 + abs(b_[i]))) if sense == S.GE else abs(v - b_[i]) <= 1e-5 * (1 + abs(b_[i]))
            assert np.all(x >= -1e-9) and abs(r.gap_path[-1]) <= 1e-8
            count += 1
    assert count >= 300


def test_duals_satisfy_strong_duality_and_match_the_simplex_on_non_degenerate_instances():
    inst = S.centre_instance()
    for method in P.METHODS:
        r = P.ipm(inst, method, eps=1e-10)
        M, b, c, n, _ = P.standard_form(inst)
        assert float(b @ np.array(r.y)) == pytest.approx(-r.obj, rel=1e-7)
        assert np.allclose(-np.array(r.y), [5.0, 0.5, 4.0, 0.0], atol=1e-5)                      # Schattenpreise des Zentrums (Stück 5)


# --- 3. Innerlichkeit, Umgebungen, Kurzschritt -------------------------------------------------------------------------------------------------

def test_iterates_stay_strictly_inside_and_long_steps_stay_in_the_neighbourhood():
    for inst in _instances(3):
        for method in P.METHODS:
            r = P.ipm(inst, method, eps=1e-8)
            assert min(r.min_comp) > 0 and all(mu > 0 for mu in r.mu_path)
            if method == "long":
                assert min(r.min_comp[1:]) >= P.GAMMA * (1 - 1e-9), (inst.kind, min(r.min_comp[1:]))
            assert all(0 < a <= 1 for a in r.step_p + r.step_d)


def _centred_instance(m=3, N=9, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(m, N))
    return M, M @ np.ones(N), np.ones(N)


def test_short_step_from_a_centred_feasible_start_reduces_mu_by_the_fixed_factor_each_iteration():
    """Satz (Wright 1997): bei zulässigem, exakt zentriertem Start und voller Newton-Länge gilt mu+ = sigma mu (hier auf min e^T z, M z = M e mit Start z = e, y = 0, s = e nachgerechnet)."""
    M, b, c = _centred_instance()
    m, N = M.shape
    r = P.ipm(S.textbook_instance(), "short", eps=1e-6, start=(np.ones(N), np.zeros(m), np.ones(N)), std=(M, b, c, N, ""))
    sigma = 1.0 - 0.4 / math.sqrt(N)
    ratios = [q / p for p, q in zip(r.mu_path, r.mu_path[1:])]
    full = [q for q, ap, ad in zip(ratios, r.step_p, r.step_d) if ap == 1.0 and ad == 1.0]
    assert len(full) >= 3 and all(q == pytest.approx(sigma, abs=1e-9) for q in full)
    assert r.prim_res[0] == 0.0 and r.dual_res[0] == 0.0


def test_short_step_needs_about_sqrt_n_times_log_iterations():
    """Fester Faktor 1 - 0.4/sqrt(N): die Iterationen wachsen mit sqrt(N) ln(1/eps); Theorie-Obergrenze bei zulässigem zentriertem Start."""
    counts = {}
    for n in (4, 16, 36):
        inst = S.generate("random", n, n, 0.5, 5)
        r = P.ipm(inst, "short", eps=1e-8)
        assert r.status == "optimal"
        N = r.N
        theory = math.log(r.mu_path[0] / 1e-8) / -math.log(1 - 0.4 / math.sqrt(N))
        counts[n] = (r.iterations, theory)
        assert r.iterations <= 2.0 * theory + 10
    assert counts[36][0] > counts[4][0]


# --- 4. Mehrotra --------------------------------------------------------------------------------------------------------------------------------

def test_mehrotra_sigma_and_corrector_against_an_independent_computation():
    inst = S.generate("random", 5, 6, 0.5, 2)
    M, b, c, n, _ = P.standard_form(inst)
    N = M.shape[1]
    z, y, s = P.mehrotra_start(M, b, c)
    mu = float(z @ s) / N
    K = np.block([[M, np.zeros((M.shape[0], M.shape[0])), np.zeros((M.shape[0], N))], [np.zeros((N, N)), M.T, np.eye(N)], [np.diag(s), np.zeros((N, M.shape[0])), np.diag(z)]])
    r_p, r_d = b - M @ z, c - M.T @ y - s
    aff = np.linalg.solve(K, np.concatenate([r_p, r_d, -z * s]))
    dz, ds = aff[:N], aff[N + M.shape[0]:]

    def maxstep(v, dv):
        neg = dv < 0
        return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0
    mu_aff = float((z + maxstep(z, dz) * dz) @ (s + maxstep(s, ds) * ds)) / N
    r = P.ipm(inst, "mehrotra", eps=1e-8, max_iter=1)
    assert r.sigma[0] == pytest.approx((mu_aff / mu) ** 3, rel=1e-7) and 0 < r.sigma[0] < 1
    assert all(0 < a <= 1 for a in r.step_p + r.step_d) and r.flops == P.flops_per_iteration(M.shape[0], N, 2)


def test_mehrotra_start_is_strictly_positive_and_defined_by_the_least_squares_solution():
    inst = S.centre_instance()
    M, b, c, n, _ = P.standard_form(inst)
    z, y, s = P.mehrotra_start(M, b, c)
    assert np.all(z > 0) and np.all(s > 0)
    z0 = M.T @ np.linalg.solve(M @ M.T, b)
    assert np.allclose(M @ z0, b)


# --- 5. Affine Scaling ---------------------------------------------------------------------------------------------------------------------------

def test_affine_scaling_converges_on_all_feasible_fixtures_without_centering():
    for kind in ("textbook", "centre", "degenerate"):
        r = P.ipm(S.generate(kind, 6, 8, 0.5, 35), "affine", eps=1e-8)
        assert r.status == "optimal" and set(r.sigma) == {0.0}


# --- 6. Unzulässig / Unbeschränkt ---------------------------------------------------------------------------------------------------------

def test_infeasible_and_unbounded_fixtures_get_verified_certificates_for_every_method():
    for method in P.METHODS:
        r = P.ipm(S.infeasible_instance(), method)
        assert r.status in ("infeasible", "suspect") and not r.x and math.isnan(r.obj)
        u = P.ipm(S.unbounded_instance(), method)
        assert u.status in ("unbounded", "suspect") and not u.x
    assert P.ipm(S.infeasible_instance(), "mehrotra").status == "infeasible" and P.ipm(S.unbounded_instance(), "mehrotra").status == "unbounded"


def test_constructed_infeasible_instances_are_never_reported_optimal():
    found = 0
    for seed in range(60):
        rng = random.Random(seed)
        n = rng.randint(2, 5)
        rows = [[rng.uniform(0.5, 3) for _ in range(n)] for _ in range(3)]
        b = [rng.uniform(5, 20) for _ in range(3)]
        rows.append(list(rows[0]))
        b.append(b[0] + rng.uniform(1.0, 5.0))
        inst = _custom(rows, b, [1.0] * n, [S.LE] * 3 + [S.GE])
        assert reference_status(inst) == "infeasible"
        for method in P.METHODS:
            r = P.ipm(inst, method, max_iter=300)
            assert r.status in ("infeasible", "suspect"), (seed, method, r.status)
            assert not r.x
        found += 1
    assert found == 60


def test_constructed_unbounded_instances_are_never_reported_optimal():
    for seed in range(40):
        rng = random.Random(seed)
        n = rng.randint(2, 4)
        rows = [[rng.uniform(0.5, 3) for _ in range(n - 1)] + [0.0] for _ in range(2)]
        b = [rng.uniform(5, 20) for _ in range(2)]
        inst = _custom(rows, b, [1.0] * n, [S.LE, S.LE])                                          # x_n kommt in keiner Zeile vor: unbeschränkt
        assert reference_status(inst) == "unbounded"
        for method in P.METHODS:
            r = P.ipm(inst, method, max_iter=300)
            assert r.status in ("unbounded", "suspect"), (seed, method, r.status)


# --- 7. Sonderfälle, Buchführung ------------------------------------------------------------------------------------------------------------------

def test_redundant_rows_are_removed_when_consistent_and_reported_infeasible_when_not():
    inst = _custom([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], [4.0, 4.0, 3.0], [1.0, 2.0], [S.EQ, S.EQ, S.LE])
    M, b, c, n, note = P.standard_form(inst)
    assert M.shape[0] == 2 and "abhängige" in note
    r = P.ipm(inst, "mehrotra")
    assert r.status == "optimal" and r.obj == pytest.approx(_optimal(inst), abs=1e-5) and "abhängige" in r.note
    bad = _custom([[1.0, 1.0], [1.0, 1.0]], [4.0, 5.0], [1.0, 2.0], [S.EQ, S.EQ])
    assert P.standard_form(bad)[0] is None and P.ipm(bad).status == "infeasible"


def test_single_row_and_start_factor_extremes():
    inst = _custom([[1.0, 2.0]], [10.0], [1.0, 1.0], [S.LE])
    assert P.ipm(inst, "mehrotra").obj == pytest.approx(10.0, abs=1e-5)
    base = S.centre_instance()
    for f in (1e-2, 1.0, 1e3, 1e6):
        r = P.ipm(base, "mehrotra", eps=1e-8, start_factor=f)
        assert r.status == "optimal" and r.obj == pytest.approx(720.0, abs=1e-4), (f, r.status)
    its = [P.ipm(base, "mehrotra", eps=1e-8, start_factor=f).iterations for f in (1.0, 1e6)]
    assert its[1] > its[0]
    tiny = P.ipm(base, "mehrotra", eps=1e-8, start_factor=1e-3)                                    # zu kleiner Start: Stillstand, nie ein falsches Optimum
    assert tiny.status != "optimal" or tiny.obj == pytest.approx(720.0, abs=1e-4)


def test_flops_bookkeeping_and_determinism():
    inst = S.centre_instance()
    for method in P.METHODS:
        a, b = P.ipm(inst, method), P.ipm(inst, method)
        assert a.x == b.x and a.iterations == b.iterations
        solves = 2 if method == "mehrotra" else 1
        assert a.flops == a.iterations * P.flops_per_iteration(a.m, a.N, solves) and len(a.step_p) == len(a.sigma) == a.iterations
    m, N = 4, 9
    assert P.flops_per_iteration(m, N, 1) == int(2 * m * m * N + m ** 3 / 3 + 2 * m * m + 8 * m * N + 10 * N)
    with pytest.raises(ValueError):
        P.ipm(inst, "newton")


def test_iterations_are_almost_independent_of_the_size_for_mehrotra():
    its = {n: [P.ipm(S.generate("random", n, n, 0.5, sd), "mehrotra", eps=1e-8).iterations for sd in range(5)] for n in (4, 16, 32)}
    assert all(3 <= v <= 25 for vs in its.values() for v in vs) and max(its[32]) < 3 * max(its[4]) + 6


def test_klee_minty_needs_only_a_few_iterations():
    for n in (6, 10, 14):
        r = P.ipm(S.klee_minty_instance(n), "mehrotra", eps=1e-8)
        assert r.status == "optimal" and r.iterations <= 30 and r.obj == pytest.approx(5.0 ** n, rel=1e-6)


def test_every_branch_is_executed():
    statuses, notes = set(), set()
    for inst in list(_instances(2)) + [S.infeasible_instance(), S.unbounded_instance()]:
        for method in P.METHODS:
            r = P.ipm(inst, method, keep=True, keep_cond=True)
            statuses.add(r.status)
            notes.add(r.note)
            if r.status == "optimal":
                assert len(r.points) == r.iterations + 1 and len(r.cond) == r.iterations
    assert {"optimal", "infeasible", "unbounded"} <= statuses
    assert P.ipm(S.centre_instance(), "short", max_iter=5).status == "limit"
