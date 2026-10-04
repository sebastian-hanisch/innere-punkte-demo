"""Unabhängige Orakel für den Kern: geschlossene Form (Klee-Minty: 2^n − 1 Pivots, Optimum 5^n), HiGHS (Optimalwert, Dualwerte über Dualität und Dual-Zulässigkeit) und die Gleichungen des Zentralpfads."""

import numpy as np
import pytest

import ipm_algorithm as A
import ipm_ipm as P
import ipm_scenario as S

linprog = pytest.importorskip("scipy.optimize").linprog


def _highs_opt(inst):
    Am, b, c = inst.arrays()
    ub = [i for i, s in enumerate(inst.senses) if s != S.EQ]
    eq = [i for i, s in enumerate(inst.senses) if s == S.EQ]
    sign = [1.0 if inst.senses[i] == S.LE else -1.0 for i in ub]
    res = linprog(-c, A_ub=(Am[ub] * np.array(sign)[:, None]) if ub else None, b_ub=(b[ub] * sign) if ub else None, A_eq=Am[eq] if eq else None, b_eq=b[eq] if eq else None,
                  bounds=[(0, None)] * inst.n, method="highs")
    assert res.status == 0
    return -res.fun


@pytest.mark.parametrize("n", range(2, 11))
def test_klee_minty_dantzig_needs_exactly_two_to_the_n_minus_one_pivots(n):
    inst = S.klee_minty_instance(n)
    sol = A.solve(inst)
    assert sol.status == "optimal" and sol.pivots == 2 ** n - 1                                  # Klee-Minty (Chvátal-Form): alle 2^n Ecken, der Pfad endet im Optimum
    assert sol.obj == pytest.approx(5.0 ** n, rel=1e-9)                                           # geschlossene Form: x = (0, ..., 0, 5^n)
    assert np.allclose(sol.x, [0.0] * (n - 1) + [5.0 ** n])
    r = P.ipm(inst, "mehrotra", eps=1e-9)
    assert r.status == "optimal" and r.obj == pytest.approx(5.0 ** n, rel=1e-6)


def _generated():
    for seed in range(12):
        yield S.generate("random", 3 + seed % 6, 3 + (seed * 5) % 7, 0.5, seed)
        yield S.generate("mixed", 4 + seed % 5, 3 + (seed * 3) % 6, 0.5, seed)
    yield S.textbook_instance()
    yield S.centre_instance()


def test_optimum_and_duals_of_simplex_and_every_ipm_method_against_highs():
    count = 0
    for inst in _generated():
        opt = _highs_opt(inst)
        _, b, c = inst.arrays()
        sol = A.solve(inst)
        assert sol.status == "optimal" and sol.obj == pytest.approx(opt, rel=1e-7, abs=1e-6)
        ys = [("simplex", np.array(sol.y))]
        for method in P.METHODS:
            r = P.ipm(inst, method, eps=1e-8)
            assert r.status == "optimal" and r.obj == pytest.approx(opt, rel=1e-6, abs=1e-5), (inst.kind, method)
            ys.append((method, -np.array(r.y)))                                                  # Standardform minimiert -c: Vorzeichen der Dualen umdrehen
            assert A.primal_violation(inst, r.x) <= 1e-5 * (1 + np.abs(b).max())
        for name, y in ys:                                                                       # starke Dualität und Dual-Zulässigkeit: y ist ein optimaler Dualvektor
            assert float(y @ b) == pytest.approx(opt, rel=1e-5, abs=1e-5), name
            assert A.dual_violation(inst, y) <= 1e-5 * (1 + np.abs(c).max()), name
        count += 1
    assert count >= 25


@pytest.mark.parametrize("seed", range(5))
def test_central_point_satisfies_the_central_path_equations(seed):
    inst = S.generate("random", 4, 5, 0.6, seed)
    M, b, c, _n, _ = P.standard_form(inst)
    for mu in (10.0, 1.0, 1e-2, 1e-5):
        z, y, s = P.central_point(M, b, c, mu)
        assert z.min() > 0 and s.min() > 0
        assert np.allclose(z * s, mu, rtol=1e-6) and np.allclose(M @ z, b, atol=1e-6) and np.allclose(M.T @ y + s, c, atol=1e-6)


# --- Ein Strahl allein beweist keine Unbeschränktheit: auch die leere Menge hat Richtungen --------------------------------------------------------

def _ray_instance(seed, contradiction):
    """Eine Variable mit freier Wachstumsrichtung (Spalte 0 oder negativ in den ≤-Zeilen, Zielkoeffizient > 0): ohne Widerspruch zulässig und unbeschränkt; mit einer Kopie von Zeile 0 als
    ≥-Zeile über der rechten Seite unzulässig - und trotzdem existiert der Strahl mit negativen Kosten (eine Richtung der leeren Menge)."""
    import random
    rng = random.Random(seed)
    n, m = rng.randint(3, 6), rng.randint(2, 4)
    rows = [[round(rng.uniform(0.5, 3.0), 1) for _ in range(n - 1)] + [0.0 if i == 0 else -round(rng.uniform(0.0, 2.0), 1)] for i in range(m)]
    b = [round(rng.uniform(5.0, 20.0), 1) for _ in range(m)]
    senses = [S.LE] * m
    if contradiction:
        rows.append(list(rows[0])), b.append(b[0] + round(rng.uniform(1.0, 5.0), 1)), senses.append(S.GE)
    c = [round(rng.uniform(0.5, 3.0), 1) for _ in range(n)]
    return S._inst(rows, b, c, senses, [f"x{j}" for j in range(n)], [f"r{i}" for i in range(len(rows))], "custom")


def _highs_status(inst):
    """HiGHS: "infeasible" | "unbounded" | "optimal"; die Zulässigkeit über ein Problem ohne Zielfunktion, damit der Präsolve nicht zwischen Status 2 und 3 schwankt."""
    Am, b, c = inst.arrays()
    sign = np.array([1.0 if s == S.LE else -1.0 for s in inst.senses])
    kw = dict(A_ub=Am * sign[:, None], b_ub=b * sign, bounds=[(0, None)] * inst.n, method="highs")
    if linprog(np.zeros(inst.n), **kw).status == 2:
        return "infeasible"
    return "optimal" if linprog(-c, **kw).status == 0 else "unbounded"


@pytest.mark.parametrize("method", ["mehrotra", "short", "affine"])
def test_a_ray_without_a_feasible_point_is_never_reported_unbounded(method):
    wrong = unbounded_ok = infeasible_ok = 0
    for seed in range(100):
        feas, infe = _ray_instance(seed, False), _ray_instance(seed, True)
        assert _highs_status(feas) == "unbounded" and _highs_status(infe) == "infeasible"             # Orakel HiGHS: die beiden Fälle sind wirklich verschieden
        wrong += P.ipm(infe, method, max_iter=300).status == "unbounded"
        unbounded_ok += P.ipm(feas, method, max_iter=300).status == "unbounded"
        infeasible_ok += P.ipm(infe, method, max_iter=300).status == "infeasible"
    assert wrong == 0                                                                              # alter Code: ein erheblicher Teil der unzulässigen Instanzen wurde als "unbeschränkt" bewiesen
    assert unbounded_ok >= 95 and infeasible_ok >= 90                                              # keine Verschlechterung: echte unbeschränkte Instanzen werden weiter erkannt


def test_feasibility_phase_returns_checked_certificates():
    for seed in range(40):
        for contradiction, want in ((False, "feasible"), (True, "infeasible")):
            M, b, _, _, _ = P.standard_form(_ray_instance(seed, contradiction))
            verdict, cert = P.feasibility(M, b)
            assert verdict == want
            if want == "feasible":
                assert cert.min() >= 0 and np.allclose(M @ cert, b, atol=1e-6 * (1 + np.abs(b).max()))
            else:
                assert P.dual_ray(M, b, cert)
