"""Konstanten der Demo Innere-Punkte-Verfahren: Regler-Bereiche, Stufen, feste Instanzen für die Auswertung, Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 40, 6
N_MIN, N_MAX, DEFAULT_N = 2, 40, 8
DENSITY = 0.5                                        # Dichte der Zufalls- und Mischinstanzen (fest)
SEED_MAX = 999999
DEFAULT_SEED = 35
EPS_EXPS = (2, 4, 6, 8, 10, 12, 14)                  # Genauigkeit 10^-k (relative Lücke und Residuen)
DEFAULT_EPS_I = 3                                    # 1e-8
START_FACTORS = (0.001, 0.01, 1, 100, 10 ** 4, 10 ** 6, 10 ** 8)   # Faktor auf den Mehrotra-Startpunkt (z und s)
DEFAULT_START_I = 2
SCALE_EXPS = (0, 2, 4, 6, 8, 10, 12, 14, 16)         # Spaltenskalierung 10^k (wie im Ellipsoid-Stück)
DEFAULT_SCALE_I = 0
METHOD_LABELS = {"affine": "Affine Scaling (σ = 0, ohne Zentrierung)", "short": "Kurzschritt (fester Faktor 1 − 0.4/√N)", "long": "Langschritt (σ = 0.1, Umgebung N₋∞)", "mehrotra": "Mehrotra (Prädiktor-Korrektor)"}
METHOD_SHORT = {"affine": "Affine Scaling", "short": "Kurzschritt", "long": "Langschritt", "mehrotra": "Mehrotra"}
STEPS = {1: "1 · Der Pfad", 2: "2 · Iterationen", 3: "3 · Gegen Simplex und Ellipsoid", 4: "4 · Grenzen"}
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_SIZES = (2, 4, 8, 16, 24, 32, 40)              # Zufallsinstanzen m = n
THEORY_SIZES = (4, 8, 16, 32)
CUBE_SIZES = tuple(range(2, 15))
EPS_SWEEP_EXPS = (2, 4, 6, 8, 10, 12, 13, 14)
PATH_POINTS = 60                                     # Punkte des Zentralpfads im 2D-Bild
_BASE = {"kind": "centre", "m": 6, "n": 8, "seed": 35, "method": "mehrotra", "eps": 3, "start": 2, "scale": 0, "step": 1}
PRESETS = {
    "Lehrbuch: der Weg durch das Innere": {**_BASE, "kind": "textbook", "step": 1, "iter_k": 3},
    "Zentrum: wenige Iterationen, teure Schritte": {**_BASE, "step": 3},
    "Iterationen fast unabhängig von n": {**_BASE, "kind": "random", "m": 16, "n": 16, "step": 2},
    "Vier Verfahren im Vergleich": {**_BASE, "kind": "random", "m": 16, "n": 16, "step": 2},
    "Kurzschritt gegen die Theorie": {**_BASE, "kind": "random", "m": 16, "n": 16, "method": "short", "step": 2},
    "Klee-Minty-Würfel: hier gewinnt das Innere": {**_BASE, "kind": "klee_minty", "n": 14, "step": 3},
    "Zufall: der Simplex bleibt vorn": {**_BASE, "kind": "random", "m": 40, "n": 40, "step": 3},
    "Startpunkt zu klein": {**_BASE, "kind": "random", "m": 16, "n": 16, "start": 0, "step": 2},
    "Genauigkeitsgrenze": {**_BASE, "kind": "random", "m": 16, "n": 16, "eps": 6, "step": 4},
    "Schlechte Skalierung": {**_BASE, "kind": "random", "m": 8, "n": 8, "scale": 6, "step": 4},
    "Unzulässig: Farkas-Strahl": {**_BASE, "kind": "infeasible", "step": 1},
    "Unbeschränkt: Strahl": {**_BASE, "kind": "unbounded", "step": 1},
}
PRESET_HELP = {
    "Lehrbuch: der Weg durch das Innere": "Nach 4 Iterationen ist das Optimum 36 bis ε = 10^-8 gefunden (der Simplex braucht 2 Pivots). Der Startpunkt (3.41, 5.79) liegt außerhalb der zulässigen Menge; schon die erste Iterierte (2.07, 5.71) ist nahe der Ecke (2, 6), und μ fällt von 4.4 über 0.39 auf 0.002.",
    "Zentrum: wenige Iterationen, teure Schritte": "7 Iterationen gegen 4 Pivots des Simplex; im Operationsmodell 5257 gegen 400 Operationen (13-fach): wenige Schritte, aber jeder löst ein dichtes Gleichungssystem.",
    "Iterationen fast unabhängig von n": "Mehrotra braucht bei ε = 10^-8 im Median 4 Iterationen bei n = 2, 8 bei n = 16 und 9 bei n = 40: die Größe wächst um das 20-Fache, die Iterationen um das 2.25-Fache. Diese Instanz (16 × 16, Seed 35): 8 Iterationen. Klick auf 'Iterationen über die Größe n berechnen'.",
    "Vier Verfahren im Vergleich": "Auf dieser Instanz (16 × 16, Seed 35): Mehrotra 8, Affine Scaling 16, Langschritt 17 und Kurzschritt 275 Iterationen; im Operationsmodell 185.512 / 362.832 / 385.509 / 6.236.175 Operationen.",
    "Kurzschritt gegen die Theorie": "Der Kurzschritt (fester Faktor 1 − 0.4/√N) braucht bei 16 × 16 275 Iterationen. Die Schranke ln(μ₀/ε) / −ln(1 − 0.4/√N) erlaubt im Median 315: gemessen sind es bei n = 16 87 % davon, bei n = 4 84 % und bei n = 32 89 %. Klick auf 'Kurzschritt gegen die Theorie prüfen'.",
    "Klee-Minty-Würfel: hier gewinnt das Innere": "Würfel mit n = 14: der Simplex (Dantzig-Regel) braucht 16383 Pivots, Mehrotra 17 Iterationen; im Operationsmodell 273.530 gegen 14.253.210 Operationen (Verhältnis 0.019). Der Kreuzungspunkt liegt bei n = 7 (Verhältnis 1.39 bei n = 6, 0.85 bei n = 7); Steepest Edge löst den Würfel dagegen in einem Pivot (Stück 3). Klick auf 'Auf dem Würfel vergleichen'.",
    "Zufall: der Simplex bleibt vorn": "Zufall 40 × 40 (Seed 35): 9 Iterationen gegen 27 Pivots, im Modell 2.791.197 gegen 179.334 Operationen (15.6-fach). Über die Größen n = 2 bis 40 liegt das Verhältnis bei 16 bis 24. Klick auf 'Operationen über die Größe berechnen'.",
    "Startpunkt zu klein": "Zufall 16 × 16, Startpunkt mit Faktor 0.001: nach 15 Iterationen bricht das Verfahren ab und meldet den Verdacht auf Unbeschränktheit, obwohl die Instanz beschränkt ist. Über fünf Instanzen erreicht Mehrotra bei 0.001 0 von 5, bei 0.01 alle 5 (25 statt 8 Iterationen) und bei 10^8 alle 5 (16 Iterationen); der Langschritt braucht bei 0.01 im Median 169 Iterationen und schafft 3 von 5. Klick auf 'Startpunkt variieren'.",
    "Genauigkeitsgrenze": "Zufall 16 × 16, ε = 10^-14: nach 442 Iterationen scheitert die Numerik (die Iterierten bleiben nicht mehr strikt positiv), es gibt kein Ergebnis; die Bruchstelle hängt von der Plattform ab (Wert vom Entwicklungsrechner unter Windows). Über fünf Instanzen erreichen bis ε = 10^-13 alle das Zertifikat (im Median 5 Iterationen bei 10^-2, 10 bei 10^-13), bei 10^-14 nur 4 von 5. Klick auf 'Genauigkeit verschärfen'.",
    "Schlechte Skalierung": "Zufall 8 × 8 (Seed 35), Spalten über 10^12 gestreut: Mehrotra findet das Optimum nach 22 Iterationen. Über fünf Instanzen löst Mehrotra bei 10^12 4 von 5 richtig (1 Stillstand; unter Linux 5 von 5), der Kurzschritt alle 5 (aber rund 165 Iterationen), der Langschritt 0 von 5 und Affine Scaling 0 von 5 (unter Linux bis 1); falsche Optima gibt es nicht. Klick auf 'Schlechte Skalierung testen'.",
    "Unzulässig: Farkas-Strahl": "Nach 2 Iterationen ist ein Strahl y mit Mᵀy ≤ 0 und bᵀy > 0 gefunden: die Mindestmenge x1 ≥ 6 widerspricht x1 ≤ 4. Der Strahl ist nachgerechnet, also ein Beweis und kein Verdacht.",
    "Unbeschränkt: Strahl": "Nach 3 Iterationen ist ein Strahl z ≥ 0 mit Mz = 0 und cᵀz < 0 gefunden: der Zielwert wächst ohne Grenze. Anders als beim Ellipsoid (Kugelrand ohne Beweis) ist das ein Beweis.",
}


def eps_label(i):
    return f"10^-{EPS_EXPS[i]}"


def start_label(i):
    f = START_FACTORS[i]
    return "Mehrotra-Start" if f == 1 else f"{f:g}-fach"


def scale_label(i):
    return "unskaliert" if SCALE_EXPS[i] == 0 else f"über 10^{SCALE_EXPS[i]}"
