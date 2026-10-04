"""Innere-Punkte-Verfahren – durch das Innere statt am Rand entlang - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Statt von Ecke zu Ecke (Simplex) oder von außen einzuschließen (Ellipsoid) läuft das Verfahren durch das Innere der zulässigen Menge auf das Optimum zu:
ein Newton-Schritt je Iteration auf den gestörten Optimalitätsbedingungen. Die Demo zeigt den Weg, zählt die Iterationen und stellt sie dem Simplex und dem Ellipsoid gegenüber.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import ipm_constants as C
import ipm_evaluation as ev
import ipm_ipm as P
import ipm_scenario as S
from ipm_evaluation import Settings, analyse
from ipm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from ipm_visualization import (
    build_cond,
    build_convergence,
    build_cube,
    build_eps,
    build_flops,
    build_path,
    build_scale,
    build_size,
    build_start,
    build_steps,
    build_theory,
)

st.set_page_config(page_title="Innere Punkte – Sebastian Hanisch", layout="wide")


def num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{0.0 if abs(x) < 5e-13 else x:.{digits}f}"


def big(x):
    return f"{x:,.0f}".replace(",", ".")


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig (Strahl)", "unbounded": "unbeschränkt (Strahl)", "suspect": "Verdacht", "numerical": "numerisch gescheitert", "limit": "Iterationsgrenze erreicht"}

st.title("🛤️ Innere-Punkte-Verfahren – durch das Innere statt am Rand entlang")
st.markdown(
    """
**Achtes Stück der Lineare-Programmierung-Reihe.** Der Simplex läuft von Ecke zu Ecke, das Ellipsoid (Stück 7) schließt die Lösung von außen ein und braucht Tausende von Iterationen. **Karmarkar (1984)** zeigte, dass es polynomial **und** schnell geht:
das Verfahren läuft **durch das Innere** der zulässigen Menge, nie am Rand, entlang des **Zentralpfads** auf das Optimum zu. Je Iteration löst es ein lineares Gleichungssystem (die **Normalgleichungen**) und macht einen Newton-Schritt auf den gestörten
Optimalitätsbedingungen x·s = μ. Vier Fragen, alle gemessen: **(1) Der Pfad** - wie sieht der Weg aus? **(2) Iterationen** - wie viele braucht es, und wovon hängt es ab? **(3) Gegen Simplex und Ellipsoid** - was kostet es, wo gewinnt es?
**(4) Grenzen** - Genauigkeit, Startpunkt, Skalierung.
"""
)
st.caption("Kind des [Ellipsoids](https://github.com/sebastian-hanisch/ellipsoid-demo) und Kontrast zum [Simplex](https://github.com/sebastian-hanisch/tableau-simplex-demo). Folgestücke sind gebaut: [Präsolve und Numerik](https://github.com/sebastian-hanisch/praesolve-demo), [PDLP](https://github.com/sebastian-hanisch/pdlp-demo) und [Crossover](https://github.com/sebastian-hanisch/crossover-demo).")

with st.expander("So funktioniert ein Innere-Punkte-Verfahren", expanded=True):
    st.markdown(
        """
1. **Standardform:** Schlupf- und Überschussvariablen machen aus allen Nebenbedingungen Gleichungen M z = b mit z ≥ 0; die Dualen y und die Dualschlupfvariablen s ≥ 0 gehören dazu (M ᵀ y + s = c).
2. **Zentralpfad:** für jedes μ > 0 gibt es genau einen Punkt mit z_j s_j = μ für alle j; für μ → 0 läuft er auf das Optimum zu. Jede Iteration zielt auf einen Punkt mit kleinerem μ.
3. **Newton-Schritt:** die Richtung löst M Δz = Residuum, Mᵀ Δy + Δs = Residuum, S Δz + Z Δs = σμ − z s. Eliminiert man Δz und Δs, bleiben die **Normalgleichungen** M (Z/S) Mᵀ Δy = rechte Seite: ein dichtes System mit Cholesky-Zerlegung, je Iteration der Hauptaufwand.
4. **Schrittlänge:** nur so weit, dass z und s strikt positiv bleiben. Die Verfahren unterscheiden sich in σ (wie stark zentriert wird) und der Schrittlänge: **Affine Scaling** (σ = 0), **Kurzschritt** (fester Faktor, Theorie √N · ln(1/ε)), **Langschritt** (σ = 0.1 und Umgebung des Pfades), **Mehrotra** (Prädiktor-Korrektor: σ aus einem Probeschritt).
5. **Start ohne Zulässigkeit:** das Verfahren braucht keinen zulässigen Startpunkt; die Residuen werden mit den Schritten kleiner. Unzulässige und unbeschränkte Instanzen zeigen sich durch Strahlen (Farkas), die die Demo nachrechnet; bei Unbeschränktheit prüft sie zusätzlich getrennt, dass es überhaupt einen zulässigen Punkt gibt (ein Strahl allein genügt nicht).
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:8], preset_names[8:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Zentrum und entartete Ecke sind fest; Zufall (≤-Ressourcen) und Mischung (mit ≥ und =) sind regelbar; der Klee-Minty-Würfel hat nur die Größe n. Unzulässig und Unbeschränkt zeigen die Strahlen.")
    cube = kind in S.CUBE_KINDS
    random_kind = kind not in S.FIXTURE_KINDS and not cube
    if random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen.")
    else:
        m = C.DEFAULT_M
    if random_kind or cube:
        cap = S.CUBE_MAX if cube else C.N_MAX
        n = st.slider("Dienste n", C.N_MIN, cap, value=min(int(ss["n_slider"]), cap), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen (beim Würfel auch die Zahl der Ressourcen).")
    else:
        n = C.DEFAULT_N
    if random_kind:
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED
    method = st.radio("Verfahren", options=list(P.METHODS), format_func=lambda v: C.METHOD_LABELS[v], key="method_select", help="Alle vier lösen dasselbe Newton-System; sie unterscheiden sich in der Zentrierung σ und der Schrittlänge.")
    eps_i = st.select_slider("Genauigkeit ε", options=list(range(len(C.EPS_EXPS))), format_func=C.eps_label, key="eps_select", help="Relative Lücke und Residuen, bei denen das Verfahren stoppt.")
    start_i = st.select_slider("Startpunkt", options=list(range(len(C.START_FACTORS))), format_func=C.start_label, key="start_select", help="Faktor auf den Startpunkt nach Mehrotra (z und s).")
    scale_i = st.select_slider("Skalierung der Spalten", options=list(range(len(C.SCALE_EXPS))), format_func=C.scale_label, key="scale_select",
                               help="Die Dienste werden mit Faktoren zwischen 1 und 10^k umgerechnet (Optimalwert gleich, Zahlen sehr verschieden groß): ein Test für die Numerik.")

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "seed_input": int(ss["seed_input"]), "method_select": method, "eps_select": int(eps_i), "start_select": int(start_i),
                   "scale_select": int(scale_i), "ipm_step": int(ss["ipm_step"])})

settings = Settings(kind, int(m), int(n), int(seed), method, int(eps_i), int(start_i), int(scale_i))
with st.spinner("Rechne..."):
    a = analyse(settings)
res, inst = a.res, a.inst

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Durch das Innere zum Optimum")
step = st.select_slider("Schritt", options=list(C.STEPS), key="ipm_step", format_func=lambda s: C.STEPS[s])

if res.status == "optimal":
    rel = abs(a.obj_error) / (1.0 + abs(a.ref_obj)) if math.isfinite(a.obj_error) else float("nan")
    if math.isfinite(rel) and rel > 1e-3:
        st.warning(f"⚠️ Das Verfahren meldet ein Optimum ({num(res.obj)}), aber der Optimalwert weicht um {rel:.1%} vom Referenzwert ({num(a.ref_obj)}) ab - die schlechte Skalierung hat die Toleranzen unterlaufen.")
    else:
        st.success(f"✅ Optimum **{num(res.obj)}** nach **{res.iterations}** Iterationen ({C.METHOD_SHORT[method]}, ε = {settings.eps:g}); Simplex: {num(a.ref_obj)} nach {a.simplex.pivots} Pivots.")
elif res.status in ("infeasible", "unbounded"):
    st.info(f"Die Instanz ist **{'unzulässig' if res.status == 'infeasible' else 'unbeschränkt'}**: {res.note} (nach {res.iterations} Iterationen). Der Strahl ist nachgerechnet und ein Beweis, kein Verdacht.")
elif res.status == "suspect":
    st.warning(f"⚠️ {res.note}. Der Simplex meldet: {a.simplex.status}. Ohne Strahl ist das kein Beweis; es kann auch ein zu kleiner Startpunkt oder eine schlechte Skalierung sein.")
elif res.status == "numerical":
    st.error(f"❌ Numerisch gescheitert nach {res.iterations} Iterationen: {res.note}.")
else:
    st.warning(f"Iterationsgrenze ({P.MAX_ITER}) erreicht ohne Zertifikat.")
if res.note and res.status == "optimal":
    st.caption(f"Hinweis: {res.note}.")

if step == 1:
    total = res.iterations
    if inst.n == 2 and res.points:
        if "iter_k" in ss:
            ss["iter_k"] = min(max(0, int(ss["iter_k"])), total)
        k = st.slider("Iteration", 0, total, key="iter_k", help="0 = Startpunkt; danach Newton-Schritt für Newton-Schritt.") if total > 0 else 0
        mu_k = res.mu_path[k] if k < len(res.mu_path) else res.mu_path[-1]
        step_txt = f" Schrittlänge primal {num(res.step_p[k], 2)}, σ = {num(res.sigma[k], 3)}." if k < len(res.step_p) else " Endpunkt."
        st.markdown(f"**Iteration {k} von {total}:** Komplementarität μ = {mu_k:.3g}, primales Residuum {res.prim_res[min(k, len(res.prim_res) - 1)]:.2g}, relative Lücke {res.gap_path[min(k, len(res.gap_path) - 1)]:.2g}.{step_txt}")
        st.plotly_chart(build_path(a, k, ev.central_path(settings)), width="stretch", key=f"s1_path_{k}")
        st.caption("Orange: die Iterierten des Verfahrens (nie auf dem Rand); grau gestrichelt: der Zentralpfad, die Kurve der Punkte mit gleichem Produkt z·s für fallendes μ; rot: die aktuelle Iterierte; Stern: Optimum des Simplex. Der Startpunkt liegt nicht zwingend in der zulässigen Menge: die Residuen werden mit den Schritten null.")
    else:
        st.markdown("Ab drei Diensten gibt es kein Bild des Wegs; die Kurven zeigen ihn: Komplementarität und Residuen fallen gleichmäßig.")
    if res.mu_path:
        st.plotly_chart(build_convergence(res), width="stretch", key="s1_conv")
        st.caption("Logarithmische Achse: μ, Lücke und Residuen fallen fast geradlinig, also um einen festen Faktor je Iteration - die Iterationen wachsen nur mit der Zahl der Stellen.")
    if res.step_p:
        st.plotly_chart(build_steps(res), width="stretch", key="s1_steps")
        st.caption("Schrittlängen (primal und dual) und Zentrierung σ je Iteration: Mehrotra wählt σ selbst, die anderen Verfahren fest.")
elif step == 2:
    tab = ev.method_table(settings)
    st.markdown("**Alle vier Verfahren auf dieser Instanz:**")
    st.dataframe(pd.DataFrame([{"Verfahren": C.METHOD_SHORT[t["method"]], "Ergebnis": STATUS_TEXT[t["status"]], "Iterationen": t["iterations"], "Optimalwert": num(t["obj"]) if math.isfinite(t["obj"]) else "-",
                                 "Operationen": big(t["flops"])} for t in tab]), hide_index=True, width="stretch")
    st.markdown("**Wovon hängt die Zahl der Iterationen ab?** (🔬 auf Abruf; Zufallsinstanzen, Median über fünf feste Instanzen)")
    tok_size = (settings.method, settings.eps_i, settings.start_i)
    if st.button("Iterationen über die Größe n berechnen", key="size_start"):
        ss["size_done"] = tok_size
    if ss.get("size_done") == tok_size:
        with st.spinner("Rechne..."):
            rows = ev.size_sweep(settings)
        st.plotly_chart(build_size(rows), width="stretch", key="s2_size")
        st.dataframe(pd.DataFrame([{"n": r["n"], **{C.METHOD_SHORT[m]: f"{r['iterations'][m]:.0f}" for m in P.METHODS}} for r in rows]), hide_index=True, width="stretch")
        st.caption("Logarithmische Achse. Die Iterationen wachsen kaum mit n; nur der Kurzschritt (fester Faktor) wächst wie √N.")
    tok_th = (settings.eps_i,)
    if st.button("Kurzschritt gegen die Theorie prüfen", key="theory_start"):
        ss["theory_done"] = tok_th
    if ss.get("theory_done") == tok_th:
        with st.spinner("Rechne..."):
            rows = ev.theory_compare(settings)
        st.plotly_chart(build_theory(rows), width="stretch", key="s2_theory")
        st.dataframe(pd.DataFrame([{"n": r["n"], "gemessen": f"{r['iterations']:.0f}", "Schranke": f"{r['theory']:.0f}", "Anteil": f"{r['share']:.0%}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Die Theorie (Wright 1997) verlangt bei zulässigem, zentriertem Start ln(μ₀/ε) / −ln(1 − 0.4/√N) Iterationen; hier mit Start ohne Zulässigkeit, und das Verfahren liegt knapp darunter.")
elif step == 3:
    st.markdown("**Innere Punkte gegen Simplex auf dieser Instanz** (Operationsmodell, kein Wandzeit-Vergleich):")
    c1, c2, c3 = st.columns(3)
    c1.metric("Iterationen / Pivots", f"{res.iterations} / {a.simplex.pivots}", delta=C.METHOD_SHORT[method] + " / Simplex", delta_color="off")
    c2.metric("Operationen Innere Punkte", big(a.flops_ipm), delta="Modell", delta_color="off")
    c3.metric("Operationen Simplex", big(a.flops_simplex), delta="Modell", delta_color="off")
    if a.flops_simplex and res.status == "optimal":
        ratio = a.flops_ipm / a.flops_simplex
        st.markdown(f"Das Verfahren braucht auf dieser Instanz das **{ratio:.2f}-Fache** der Operationen des dichten Tableau-Simplex." + (" Der Simplex ist hier billiger." if ratio > 1 else " Das Innere ist hier billiger."))
    st.caption("Modell: eine Iteration kostet 2m²N (Normalmatrix) + m³/3 (Cholesky) + 2m² je rechte Seite + 8mN + 10N Operationen; ein Pivot des dichten Tableaus 2(m+1)(Spalten+1). Das Ellipsoid aus Stück 7 braucht auf denselben Zufallsinstanzen Tausende von Iterationen (etwa 25 n²).")
    tok = (settings.method, settings.eps_i, settings.start_i)
    if st.button("Operationen über die Größe berechnen", key="flops_start"):
        ss["flops_done"] = tok
    if ss.get("flops_done") == tok:
        with st.spinner("Rechne..."):
            rows = ev.size_sweep(settings)
        st.plotly_chart(build_flops(rows, method), width="stretch", key="s3_flops")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Iterationen": f"{r['iterations'][method]:.0f}", "Pivots": f"{r['pivots']:.0f}", "Operationen Innere Punkte": big(r["flops_ipm"]), "Operationen Simplex": big(r["flops_simplex"]),
                                     "Verhältnis": f"{r['ratio']:.1f}"} for r in rows]), hide_index=True, width="stretch")
        st.caption("Zufallsinstanzen m = n, Median über fünf feste Instanzen. Beide Verfahren wachsen in diesem Modell ähnlich mit n; der Abstand bleibt.")
    st.markdown("**Der schlimmste Fall des Simplex:** der Klee-Minty-Würfel aus Stück 3 (🔬 auf Abruf).")
    if st.button("Auf dem Würfel vergleichen", key="cube_start"):
        ss["cube_done"] = tok
    if ss.get("cube_done") == tok:
        with st.spinner("Rechne..."):
            cs = ev.cube_sweep(settings)
        st.plotly_chart(build_cube(cs, method), width="stretch", key="s3_cube")
        st.dataframe(pd.DataFrame([{"n": r["n"], "Iterationen": r["iterations"], "Pivots (Dantzig)": big(r["pivots"]), "Operationen Innere Punkte": big(r["flops_ipm"]), "Operationen Simplex": big(r["flops_simplex"]),
                                     "Verhältnis": num(r["ratio"])} for r in cs["rows"]]), hide_index=True, width="stretch")
        st.caption((f"Ab n = {cs['crossover']} braucht das Verfahren weniger Operationen als der Simplex mit der Dantzig-Regel" if cs["crossover"] else "In diesem Bereich braucht das Verfahren immer mehr Operationen als der Simplex")
                   + ": der Simplex besucht alle 2^n Ecken, die Iterationen der Inneren Punkte wachsen kaum. Gegen Steepest Edge (ein Pivot, Stück 3) gäbe es keinen Kreuzungspunkt.")
else:
    st.markdown("**Wie gut konditioniert bleiben die Normalgleichungen?** Die Matrix M D Mᵀ ist gegen Ende der Iteration am schlechtesten konditioniert, weil D = Z/S extreme Werte annimmt:")
    if res.cond:
        st.plotly_chart(build_cond(res), width="stretch", key="s4_cond")
        st.caption("Bei ε = 10^-8 bleibt die Kondition auf Zufallsinstanzen im Bereich von etwa 10^4 bis 10^7; erst nahe der Rundungsgenauigkeit (ε = 10^-14) steigt sie auf manchen Instanzen auf 10^15 und mehr (in den fünf Sweep-Instanzen auf einer; die anderen bleiben unter 10^10), und dort kommt es je nach Instanz und Plattform zum Abbruch.")
    tok_eps = (settings.n, settings.method, settings.start_i)
    if st.button("Genauigkeit verschärfen", key="eps_start"):
        ss["eps_done"] = tok_eps
    if ss.get("eps_done") == tok_eps:
        with st.spinner("Rechne..."):
            rows = ev.eps_sweep(settings)
        st.plotly_chart(build_eps(rows), width="stretch", key="s4_eps")
        st.dataframe(pd.DataFrame([{"ε": f"10^-{r['k']}", "Iterationen (Median)": f"{r['iterations']:.0f}", "Läufe mit Zertifikat": f"{r['optimal']} von 5"} for r in rows]), hide_index=True, width="stretch")
        st.caption(f"Zufallsinstanzen m = n = {settings.n}. Jede zusätzliche Stelle kostet nur ein bis zwei Iterationen; unterhalb der Rundungsgrenze scheitert das Verfahren.")
    tok_start = (settings.n, settings.eps_i)
    if st.button("Startpunkt variieren", key="start_start"):
        ss["start_done"] = tok_start
    if ss.get("start_done") == tok_start:
        with st.spinner("Rechne..."):
            rows = ev.start_sweep(settings)
        st.plotly_chart(build_start(rows), width="stretch", key="s4_start")
        st.caption(f"Zufallsinstanzen m = n = {settings.n}. Große Startpunkte kosten nur wenige Iterationen mehr; zu kleine lassen das Verfahren stillstehen (das Ergebnis ist dann ein Verdacht, kein Optimum).")
    tok_scale = ()
    if st.button("Schlechte Skalierung testen", key="scale_start"):
        ss["scale_done"] = tok_scale
    if ss.get("scale_done") == tok_scale:
        with st.spinner("Rechne..."):
            rows = ev.scale_sweep(settings)
        st.plotly_chart(build_scale(rows), width="stretch", key="s4_scale")
        st.dataframe(pd.DataFrame([{"Spalten über": f"10^{r['k']}", **{C.METHOD_SHORT[m]: f"{r[m]['right']} richtig, {r[m]['other']} Stillstand, {r[m]['wrong']} falsch" for m in P.METHODS}} for r in rows]), hide_index=True, width="stretch")
        st.caption("Fünf feste 8 × 8-Instanzen, deren Spalten mit Faktoren zwischen 1 und 10^k umgerechnet sind (der Optimalwert bleibt gleich). 'Richtig': relativer Fehler unter 0.1 %; 'Stillstand': Verdacht, Abbruch oder Grenze.")

st.markdown("---")
st.markdown("## ⚙️ Der gewählte Fall")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Iterationen", str(res.iterations), delta=STATUS_TEXT[res.status], delta_color="off")
m2.metric("Pivots (Simplex)", str(a.simplex.pivots), delta=a.simplex.status, delta_color="off")
m3.metric("Operationen Innere / Simplex", f"{a.flops_ipm / a.flops_simplex:,.1f}×".replace(",", ".") if a.flops_simplex and a.flops_ipm else "-", delta="Modell", delta_color="off")
m4.metric("Ergebnis", num(res.obj) if res.x else "-", delta=("Referenz " + num(a.ref_obj)) if math.isfinite(a.ref_obj) else "keine Referenz", delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Innere Punkte sind immer schneller.** | Im Operationsmodell liegt das Verfahren auf Zufallsinstanzen bis n = 40 beim etwa 16- bis 24-Fachen des Simplex (wenige Pivots gegen wenige Iterationen mit teuren Schritten); nur auf dem Klee-Minty-Würfel gewinnt es, ab n = 7. | Große, dünne LPs |
| **Dichte Rechnung genügt.** | Die Demo faktorisiert die Normalmatrix dicht (m³/3). Echte Löser nutzen dünne Faktorisierungen, Präsolve und eine besondere Behandlung dichter Spalten; erst dort spielt das Verfahren seine Stärke aus. | Dünne Lineare Algebra |
| **Der Startpunkt ist egal.** | Große Startpunkte kosten kaum etwas, zu kleine lassen das Verfahren stillstehen. Ein "Verdacht" bei Unzulässigkeit oder Unbeschränktheit ist kein Beweis; belegt ist nur ein nachgerechneter Strahl (bei Unbeschränktheit samt zulässigem Punkt). | Homogene selbstduale Einbettung |
| **Das Ergebnis ist eine Ecke.** | Das Verfahren endet im Inneren, nahe am Optimum, nicht in einer Ecke; Duale und Basis (Ranging, Warmstart) bekommt man erst mit **Crossover**. | Crossover |
| **Gleitkomma ist genug.** | Die Normalmatrix wird gegen Ende extrem schlecht konditioniert; schlechte Skalierung der Spalten kostet Iterationen und kann zum Stillstand führen, und ab etwa ε = 10^-14 lässt sich das Zertifikat nicht mehr schließen. | Präsolve und Skalierung |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Standardform:** $\min c^\top z$ unter $Mz = b$, $z \ge 0$; dual $\max b^\top y$ unter $M^\top y + s = c$, $s \ge 0$. **Zentralpfad:** $Mz = b$, $M^\top y + s = c$, $z_j s_j = \mu$. **Newton-Schritt** (Residuen $r_p = b - Mz$, $r_d = c - M^\top y - s$, $r_c = \sigma\mu e - Zs$):
$M \Delta z = r_p$, $M^\top \Delta y + \Delta s = r_d$, $S\Delta z + Z\Delta s = r_c$. **Normalgleichungen:** $M (S^{-1}Z) M^\top \Delta y = r_p + M(S^{-1}Z r_d - S^{-1} r_c)$, danach $\Delta s = r_d - M^\top\Delta y$ und $\Delta z = S^{-1}(r_c - Z\Delta s)$.
**Kurzschritt:** $\sigma = 1 - 0.4/\sqrt N$, in der Theorie $O(\sqrt N \ln(1/\varepsilon))$ Iterationen. **Mehrotra:** Probeschritt mit $\sigma = 0$, dann $\sigma = (\mu_\mathrm{aff}/\mu)^3$ und der Korrektor $\Delta z_\mathrm{aff}\Delta s_\mathrm{aff}$ in $r_c$. **Zertifikate:** ein $y$ mit $M^\top y \le 0$, $b^\top y > 0$ beweist die Unzulässigkeit (Farkas), ein $z \ge 0$ mit $Mz = 0$, $c^\top z < 0$ die Unbeschränktheit.

**Literatur.** Karmarkar, N. (1984). *A new polynomial-time algorithm for linear programming.* Combinatorica 4(4), 373-395. Mehrotra, S. (1992). *On the implementation of a primal-dual interior point method.* SIAM Journal on Optimization 2(4), 575-601.
Wright, S. J. (1997). *Primal-Dual Interior-Point Methods.* SIAM.

Implementiert in `ipm_ipm.py` (Newton-Schritt, vier Verfahren, Zertifikate), `ipm_algorithm.py` (Tableau-Simplex als Vergleich), `ipm_evaluation.py`, `ipm_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Lineare Programmierung: vom Tableau zum Crossover](https://sebastianhanisch.net/konzepte-lineare-programmierung.html)."
)
