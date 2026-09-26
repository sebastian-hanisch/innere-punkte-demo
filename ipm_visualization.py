"""Plotly-Abbildungen: Weg durch das Innere und Zentralpfad (n = 2), Konvergenzkurven, Schrittlängen, Iterationen über n / Genauigkeit / Startpunkt, Operationen gegen den Simplex, Klee-Minty-Würfel,
Kurzschritt gegen Theorie, Skalierung, Kondition. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen; bei gleichem Achsenmaßstab (scaleanchor) gibt es keine expliziten Bereiche."""

import numpy as np
import plotly.graph_objects as go

import ipm_constants as C
import ipm_ipm as P

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"
METHOD_COLORS = {"affine": PURPLE, "short": GREY, "long": BLUE, "mehrotra": ORANGE}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _box(inst):
    """Obere Schranken je Dienst aus den <=-Zeilen mit nichtnegativen Koeffizienten; ohne Schranke das 1.5-Fache der größten."""
    A, b, _c = inst.arrays()
    U = np.full(inst.n, np.inf)
    for i, s in enumerate(inst.senses):
        if s != P.LE or np.any(A[i] < 0):
            continue
        for j in range(inst.n):
            if A[i, j] > 0:
                U[j] = min(U[j], b[i] / A[i, j])
    finite = U[np.isfinite(U)]
    fill = 1.5 * (finite.max() if len(finite) else 10.0)
    return np.where(np.isfinite(U), U, fill)


def feasible_polygon(inst):
    """Zulässiges Vieleck einer Instanz mit zwei Variablen (Kasten [0, U], an <=- und >=-Zeilen abgeschnitten; Gleichungen bleiben unberücksichtigt)."""
    U = _box(inst)
    poly = [np.array([0.0, 0.0]), np.array([U[0], 0.0]), np.array([U[0], U[1]]), np.array([0.0, U[1]])]
    A, b, _c = inst.arrays()
    for i, s in enumerate(inst.senses):
        if s == P.EQ:
            continue
        a, beta = (A[i], b[i]) if s == P.LE else (-A[i], -b[i])
        out = []
        for k, p in enumerate(poly):
            q = poly[(k + 1) % len(poly)]
            sp, sq = float(a @ p) - beta, float(a @ q) - beta
            if sp <= 0:
                out.append(p)
            if (sp < 0 < sq) or (sq < 0 < sp):
                out.append(p + (q - p) * (sp / (sp - sq)))
        poly = out
        if not poly:
            break
    return poly


def build_path(a, k, path):
    """Zwei Variablen: zulässige Menge, Zentralpfad (Kurve der Zentren für fallendes mu), bisherige Iterierte des Verfahrens, aktueller Iterierter, Höhenlinie durch das Optimum und der Optimalpunkt des Simplex."""
    inst, res = a.inst, a.res
    poly = feasible_polygon(inst)
    fig = go.Figure()
    if poly:
        px, py = [p[0] for p in poly], [p[1] for p in poly]
        fig.add_trace(go.Scatter(x=px + [px[0]], y=py + [py[0]], fill="toself", fillcolor="rgba(47,107,101,0.18)", line=dict(color=TEAL, width=2), name="zulässige Menge", hoverinfo="skip"))
    if path is not None and len(path):
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=GREY, width=3, dash="dash"), name="Zentralpfad", hoverinfo="skip"))
    pts = np.array(res.points[: k + 1])
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=7, color=ORANGE), name=f"Iterierte 0 bis {k}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[pts[-1, 0]], y=[pts[-1, 1]], mode="markers", marker=dict(size=13, color=RED, line=dict(color="white", width=1)), name="aktuelle Iterierte", hoverinfo="skip"))
    if a.simplex.status == "optimal":
        xs = a.simplex.x
        fig.add_trace(go.Scatter(x=[xs[0]], y=[xs[1]], mode="markers", marker=dict(size=14, color=BLUE, symbol="star"), name="Optimum (Simplex)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Dienst 1", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="Dienst 2")
    return _base(fig, 460, legend_y=-0.3)


def build_convergence(res):
    """mu, relative Lücke und Residuen über die Iterationen (log-y): fallen sie gleichmäßig?"""
    its = np.arange(0, len(res.mu_path))
    fig = go.Figure()
    for name, ys, color, dash in (("Komplementarität μ", res.mu_path, ORANGE, "solid"), ("relative Lücke", res.gap_path, BLUE, "solid"), ("primales Residuum", res.prim_res, TEAL, "dash"), ("duales Residuum", res.dual_res, PURPLE, "dot")):
        fig.add_trace(go.Scatter(x=its, y=np.maximum(np.array(ys, dtype=float), 1e-20), mode="lines+markers", line=dict(color=color, width=3 if name.startswith("Komp") else 2, dash=dash), name=name))
    fig.update_yaxes(type="log", title_text="Wert")
    fig.update_xaxes(title_text="Iteration", dtick=max(1, len(its) // 10))
    return _base(fig, 320, legend_y=-0.4)


def build_steps(res):
    its = np.arange(1, len(res.step_p) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=its, y=res.step_p, mode="lines+markers", line=dict(color=TEAL, width=3), name="Schrittlänge primal"))
    fig.add_trace(go.Scatter(x=its, y=res.step_d, mode="lines+markers", line=dict(color=ORANGE, width=2), name="Schrittlänge dual"))
    fig.add_trace(go.Scatter(x=its, y=res.sigma, mode="lines", line=dict(color=GREY, dash="dash"), name="Zentrierung σ"))
    fig.update_yaxes(title_text="Länge bzw. σ", range=[0, 1.05], fixedrange=True)
    fig.update_xaxes(title_text="Iteration", dtick=max(1, len(its) // 10))
    return _base(fig, 280, legend_y=-0.4)


def build_cond(res):
    its = np.arange(1, len(res.cond) + 1)
    fig = go.Figure(go.Scatter(x=its, y=res.cond, mode="lines+markers", line=dict(color=PURPLE, width=3), name="Kondition der Normalmatrix"))
    fig.update_yaxes(type="log", title_text="Kondition von M D Mᵀ")
    fig.update_xaxes(title_text="Iteration", dtick=max(1, len(its) // 10))
    return _base(fig, 280, legend_y=-0.4)


def build_size(rows):
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    for m in P.METHODS:
        fig.add_trace(go.Scatter(x=ns, y=[r["iterations"][m] for r in rows], mode="lines+markers", line=dict(color=METHOD_COLORS[m], width=3), name=C.METHOD_SHORT[m]))
    fig.update_yaxes(type="log", title_text="Iterationen (Median)")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)")
    return _base(fig, 340, legend_y=-0.4)


def build_eps(rows):
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["iterations"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Iterationen (Median)"))
    fails = [r for r in rows if r["optimal"] < 5]
    if fails:
        fig.add_trace(go.Scatter(x=[r["k"] for r in fails], y=[r["iterations"] for r in fails], mode="markers", marker=dict(size=12, color=RED, symbol="x"), name="nicht alle Läufe erreichen das Zertifikat"))
    fig.update_xaxes(title_text="Genauigkeit 10^-k", dtick=1)
    fig.update_yaxes(title_text="Iterationen")
    return _base(fig, 300, legend_y=-0.4)


def build_start(rows):
    fs = [str(r["factor"]) for r in rows]
    fig = go.Figure()
    for m in ("mehrotra", "long"):
        fig.add_trace(go.Bar(x=fs, y=[r[m]["iterations"] for r in rows], marker_color=METHOD_COLORS[m], name=C.METHOD_SHORT[m], text=[f"{r[m]['optimal']}/5" for r in rows], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Faktor auf den Startpunkt (Beschriftung: erfolgreiche Läufe)", type="category")
    fig.update_yaxes(title_text="Iterationen (Median)")
    return _base(fig, 320, legend_y=-0.4)


def build_flops(rows, method):
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_ipm"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name=C.METHOD_SHORT[method]))
    fig.add_trace(go.Scatter(x=ns, y=[max(r["flops_simplex"], 1) for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)")
    return _base(fig, 320, legend_y=-0.4)


def build_cube(sweep, method):
    rows = sweep["rows"]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_ipm"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name=C.METHOD_SHORT[method]))
    fig.add_trace(go.Scatter(x=ns, y=[r["flops_simplex"] for r in rows], mode="lines+markers", line=dict(color=TEAL, width=3), name="Simplex (Dantzig-Regel)"))
    if sweep["crossover"]:
        fig.add_vline(x=sweep["crossover"] - 0.5, line=dict(color=GREY, dash="dot"))
    fig.update_yaxes(type="log", title_text="Operationen (Modell)")
    fig.update_xaxes(title_text="Größe n des Würfels", dtick=2)
    return _base(fig, 320, legend_y=-0.4)


def build_theory(rows):
    ns = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=[r["theory"] for r in rows], marker_color="rgba(138,143,152,0.6)", name="Schranke ln(μ₀/ε) / −ln(1 − 0.4/√N)"))
    fig.add_trace(go.Bar(x=ns, y=[r["iterations"] for r in rows], marker_color=TEAL, name="gemessen (Kurzschritt)"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Größe n (Zufall, m = n)", type="category")
    fig.update_yaxes(title_text="Iterationen")
    return _base(fig, 300, legend_y=-0.4)


def build_scale(rows):
    ks = [f"10^{r['k']}" for r in rows]
    fig = go.Figure()
    for m in P.METHODS:
        fig.add_trace(go.Bar(x=ks, y=[r[m]["right"] for r in rows], marker_color=METHOD_COLORS[m], name=C.METHOD_SHORT[m]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Spalten über 10^k gestreut", type="category")
    fig.update_yaxes(title_text="richtig gelöste Instanzen (von 5)", range=[0, 5.3], fixedrange=True)
    return _base(fig, 320, legend_y=-0.4)
