"""Presets: gültige Werte und jede Zahl der Hilfetexte gegen die echten Auswertungsfunktionen (Iterationszahlen mit kleinem Band: Gleitkomma-Rundung kann sie plattformabhängig verschieben)."""

import pytest

import ipm_constants as C
import ipm_evaluation as ev
from ipm_evaluation import Settings
from ipm_presets import PRESET_KEYS, SETTING_SPECS


def near(x, want, rel=0.03, tol=2):
    return abs(x - want) <= max(rel * abs(want), tol)


def _settings(name, **over):
    p = {**C.PRESETS[name], **over}
    return Settings(p["kind"], p["m"], p["n"], p["seed"], p["method"], p["eps"], p["start"], p["scale"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 12
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "method", "eps", "start", "scale"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()
        if "iter_k" in p:
            assert p["step"] == 1 and p["iter_k"] > 0


def test_help_lehrbuch_and_zentrum():
    name = "Lehrbuch: der Weg durch das Innere"
    a = ev.analyse(_settings(name))
    assert near(a.res.iterations, 4, tol=1) and a.simplex.pivots == 2 and a.res.obj == pytest.approx(36.0, abs=1e-5)
    p0, p1 = a.res.points[0], a.res.points[1]
    assert [round(v, 2) for v in p0] == pytest.approx([3.41, 5.79], abs=0.02) and [round(v, 2) for v in p1] == pytest.approx([2.07, 5.71], abs=0.02)
    assert 3 * p0[0] + 2 * p0[1] > 18                                                                   # Startpunkt außerhalb der zulässigen Menge
    assert round(a.res.mu_path[0], 1) == pytest.approx(4.4, abs=0.1) and round(a.res.mu_path[1], 2) == pytest.approx(0.39, abs=0.02) and a.res.mu_path[2] == pytest.approx(0.002, abs=0.001)
    _has(name, "4 Iterationen", "36", "2 Pivots", "(3.41, 5.79)", "(2.07, 5.71)", "(2, 6)", "4.4", "0.39", "0.002")
    z = ev.analyse(_settings("Zentrum: wenige Iterationen, teure Schritte"))
    assert near(z.res.iterations, 7, tol=1) and z.simplex.pivots == 4 and z.flops_simplex == 400 and round(z.flops_ipm / z.flops_simplex) == pytest.approx(13, abs=1)
    _has("Zentrum: wenige Iterationen, teure Schritte", "7 Iterationen", "4 Pivots", "5257 gegen 400", "13-fach")


def test_help_size_methods_theory():
    rows = {r["n"]: r for r in ev.size_sweep(_settings("Iterationen fast unabhängig von n"))}
    assert near(rows[2]["iterations"]["mehrotra"], 4, tol=1) and near(rows[16]["iterations"]["mehrotra"], 8, tol=1) and near(rows[40]["iterations"]["mehrotra"], 9, tol=1)
    assert near(ev.analyse(_settings("Iterationen fast unabhängig von n")).res.iterations, 8, tol=1)
    _has("Iterationen fast unabhängig von n", "4 Iterationen bei n = 2", "8 bei n = 16", "9 bei n = 40", "20-Fache", "2.25-Fache", "8 Iterationen")
    tab = {t["method"]: t for t in ev.method_table(_settings("Vier Verfahren im Vergleich"))}
    assert near(tab["mehrotra"]["iterations"], 8, tol=1) and near(tab["affine"]["iterations"], 16, tol=2) and near(tab["long"]["iterations"], 17, tol=2) and near(tab["short"]["iterations"], 275, rel=0.03)
    assert near(tab["mehrotra"]["flops"], 185512, rel=0.1) and near(tab["short"]["flops"], 6236175, rel=0.05)
    _has("Vier Verfahren im Vergleich", "Mehrotra 8", "Affine Scaling 16", "Langschritt 17", "Kurzschritt 275", "185.512 / 362.832 / 385.509 / 6.236.175")
    name = "Kurzschritt gegen die Theorie"
    th = {r["n"]: r for r in ev.theory_compare(_settings(name))}
    assert near(ev.analyse(_settings(name)).res.iterations, 275, rel=0.03) and near(th[16]["theory"], 315, rel=0.03) and [round(100 * th[n]["share"]) for n in (16, 4, 32)] == pytest.approx([87, 84, 89], abs=2)
    _has(name, "275 Iterationen", "315", "87 %", "84 %", "89 %")


def test_help_cube_and_simplex_ahead():
    name = "Klee-Minty-Würfel: hier gewinnt das Innere"
    a = ev.analyse(_settings(name))
    assert a.inst.n == 14 and a.simplex.pivots == 16383 and near(a.res.iterations, 17, tol=2) and near(a.flops_ipm, 273530, rel=0.1) and a.flops_simplex == 14253210 and round(a.flops_ipm / a.flops_simplex, 3) == pytest.approx(0.019, abs=0.003)
    sw = ev.cube_sweep(_settings(name))
    r = {row["n"]: row for row in sw["rows"]}
    assert sw["crossover"] == 7 and round(r[6]["ratio"], 2) == pytest.approx(1.39, abs=0.1) and round(r[7]["ratio"], 2) == pytest.approx(0.85, abs=0.1)
    _has(name, "n = 14", "16383", "17 Iterationen", "273.530 gegen 14.253.210", "0.019", "n = 7", "1.39", "0.85", "Steepest Edge")
    name = "Zufall: der Simplex bleibt vorn"
    b = ev.analyse(_settings(name))
    assert near(b.res.iterations, 9, tol=1) and near(b.simplex.pivots, 27, tol=3) and round(b.flops_ipm / b.flops_simplex, 1) == pytest.approx(15.6, abs=1.5)
    ratios = [row["ratio"] for row in ev.size_sweep(_settings(name))]
    assert 15 <= min(ratios) and max(ratios) <= 25.5
    _has(name, "9 Iterationen gegen 27 Pivots", "2.791.197 gegen 179.334", "15.6-fach", "16 bis 24")


def test_help_start_accuracy_scaling_certificates():
    name = "Startpunkt zu klein"
    a = ev.analyse(_settings(name))
    assert a.res.status == "suspect" and near(a.res.iterations, 15, tol=4) and "Unbeschränktheit" in a.res.note
    rows = {r["factor"]: r for r in ev.start_sweep(_settings(name))}
    assert rows[0.001]["mehrotra"]["optimal"] == 0 and rows[0.01]["mehrotra"]["optimal"] == 5 and near(rows[0.01]["mehrotra"]["iterations"], 25, tol=4) and rows[1e8]["mehrotra"]["optimal"] == 5 and near(rows[1e8]["mehrotra"]["iterations"], 16, tol=3)
    assert near(rows[0.01]["long"]["iterations"], 169, rel=0.3) and rows[0.01]["long"]["optimal"] == 3
    _has(name, "0.001", "15 Iterationen", "0.001 0 von 5", "bei 0.01 alle 5 (25 statt 8", "10^8 alle 5 (16 Iterationen)", "169 Iterationen", "3 von 5")
    name = "Genauigkeitsgrenze"
    e = ev.analyse(_settings(name))
    assert e.res.status in ("numerical", "optimal") and e.res.iterations > 100                          # Windows: numerisch gescheitert nach 442 Iterationen; die Bruchstelle hängt von der Plattform ab
    rows = {r["k"]: r for r in ev.eps_sweep(_settings(name))}
    assert all(rows[k]["optimal"] == 5 for k in (2, 4, 6, 8, 10, 12, 13)) and near(rows[2]["iterations"], 5, tol=1) and near(rows[13]["iterations"], 10, tol=2) and rows[14]["optimal"] <= 5
    _has(name, "ε = 10^-14", "442 Iterationen", "bis ε = 10^-13", "5 Iterationen bei 10^-2", "10 bei 10^-13", "4 von 5")
    name = "Schlechte Skalierung"
    s = ev.analyse(_settings(name))
    assert s.res.status == "optimal" and near(s.res.iterations, 22, tol=6)
    sc = {r["k"]: r for r in ev.scale_sweep(_settings(name))}
    assert sc[12]["mehrotra"]["right"] in (4, 5) and sc[12]["short"]["right"] == 5 and sc[12]["long"]["right"] == 0 and sc[12]["affine"]["right"] in (0, 1) and near(sc[12]["short"]["iterations"], 165, rel=0.15)   # 10^12: Windows 4 / Linux 5
    _has(name, "22 Iterationen", "4 von 5 richtig (1 Stillstand", "alle 5", "165", "0 von 5", "falsche Optima gibt es nicht")
    inf, unb = ev.analyse(_settings("Unzulässig: Farkas-Strahl")), ev.analyse(_settings("Unbeschränkt: Strahl"))
    assert inf.res.status == "infeasible" and inf.res.iterations <= 4 and inf.inst.senses[2] == ">=" and unb.res.status == "unbounded" and unb.res.iterations <= 5
    _has("Unzulässig: Farkas-Strahl", "2 Iterationen", "x1 ≥ 6", "x1 ≤ 4", "nachgerechnet")
    _has("Unbeschränkt: Strahl", "3 Iterationen", "Beweis")
