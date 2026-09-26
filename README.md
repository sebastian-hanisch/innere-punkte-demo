# Innere-Punkte-Verfahren – durch das Innere statt am Rand entlang – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-innere-punkte-demo.streamlit.app/)**


Achtes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind des [Ellipsoids](https://github.com/sebastian-hanisch/ellipsoid-demo) und Kontrast zum [Simplex](https://github.com/sebastian-hanisch/tableau-simplex-demo). Der Simplex läuft von Ecke zu Ecke, das Ellipsoid schließt die Lösung von außen ein und braucht Tausende von Iterationen. **Karmarkar (1984)** zeigte, dass es polynomial **und** schnell geht: das Verfahren läuft **durch das Innere** der zulässigen Menge, nie am Rand, entlang des **Zentralpfads** auf das Optimum zu. Je Iteration löst es ein lineares Gleichungssystem (die **Normalgleichungen**) und macht einen Newton-Schritt auf den gestörten Optimalitätsbedingungen z_j · s_j = μ. Die Demo implementiert vier primal-duale Verfahren (Affine Scaling, Kurzschritt, Langschritt, Mehrotra) mit Start ohne Zulässigkeit und **misst**, was davon stimmt. Vier Fragen: **(1) Der Pfad** – wie sieht der Weg aus? **(2) Iterationen** – wie viele braucht es, und wovon hängt es ab? **(3) Gegen Simplex und Ellipsoid** – was kostet es, wo gewinnt es? **(4) Grenzen** – Genauigkeit, Startpunkt, Skalierung.

**Einordnung in die Reihe:** geplant sind zwölf Stücke, dies ist das achte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo]  →  [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [gebaut: lp-dualitaet-demo, dualer-simplex-demo]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [gebaut: ellipsoid-demo]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP
      [DIESES STÜCK]      →       [nicht gebaut]            →       [nicht gebaut]
```

Ergebnis in Kürze: **Wenige Iterationen, fast unabhängig von der Größe – aber im Operationsmodell auf Zufallsinstanzen trotzdem etwa das 16- bis 24-Fache des Simplex; nur auf dem Klee-Minty-Würfel gewinnt das Innere, und dort schon ab n = 7 (das Ellipsoid erst ab n = 13).** Mehrotra braucht bei ε = 10⁻⁸ im Median **4 Iterationen bei n = 2 und 9 bei n = 40** (Langschritt 11 und 21, Affine Scaling 10 und 19), der Kurzschritt mit festem Faktor wächst dagegen wie √N (79 und 448) und liegt bei 84 bis 89 % der Theorie-Schranke. Die Zahl der Stellen kostet fast nichts: von ε = 10⁻² bis 10⁻¹³ wachsen die Iterationen von 5 auf 10. Auf dem **Klee-Minty-Würfel** (n = 14) braucht der Simplex mit Dantzig-Regel 16383 Pivots, Mehrotra 17 Iterationen (273.530 gegen 14.253.210 Operationen im Modell). **Grenzen:** zu kleine Startpunkte lassen das Verfahren stillstehen (Faktor 0.001: 0 von 5 Läufen), bei schlechter Skalierung der Spalten (10¹²) jammt der Langschritt in allen Läufen, während Mehrotra 4 bis 5 von 5 richtig löst (je nach Plattform) und der konservative Kurzschritt alle 5; ab etwa ε = 10⁻¹⁴ lässt sich das Zertifikat nicht mehr verlässlich schließen. Unzulässigkeit und Unbeschränktheit werden über **nachgerechnete Strahlen** bewiesen (Farkas), nicht nur vermutet.

| Frage | Ergebnis (Auslastungsplanung als Standard-LP max c·x, Start ohne Zulässigkeit; Zufallsinstanzen m = n, Median über 5 feste Instanzen, Seeds 100000–100004; Operationen im Modell, keine Wandzeit; vollständig deterministisch) |
|---|---|
| **Stimmt das Verfahren?** | ✅ Newton-Richtung gleich der Lösung des vollen KKT-Systems (1e-9); Optimum gleich HiGHS und Simplex auf über 300 Läufen (Zufall, Mischung mit ≥ und =, Zentrum, Lehrbuch, entartet, Würfel n ≤ 14, alle vier Verfahren bis ε = 10⁻⁸); starke Dualität, Schattenpreise des Zentrums gleich Stück 5 (5, 0.5, 4, 0); Iterierte immer strikt positiv, Langschritt in seiner Umgebung; Kurzschritt reduziert μ bei zentriertem, zulässigem Start exakt um den festen Faktor 1 − 0.4/√N; Mehrotra-σ gleich unabhängiger Neuberechnung; 60 konstruierte unzulässige und 40 unbeschränkte Instanzen enden nie als Optimum |
| **Lehrbuch von Hand** | **4 Iterationen** bis 10⁻⁸ (der Simplex: 2 Pivots). Der Startpunkt (3.41, 5.79) liegt außerhalb der zulässigen Menge, schon die erste Iterierte (2.07, 5.71) ist nahe der Ecke (2, 6); μ fällt von 4.4 über 0.39 auf 0.002. Zentrum: 7 Iterationen gegen 4 Pivots, 5257 gegen 400 Operationen (13-fach) |
| **Iterationen über n** | ε = 10⁻⁸, n = 2 / 4 / 8 / 16 / 24 / 32 / 40: **Mehrotra 4 / 5 / 6 / 8 / 8 / 9 / 9**, Langschritt 11 / 12 / 14 / 17 / 19 / 21 / 21, Affine Scaling 10 / 11 / 13 / 16 / 19 / 19 / 19, Kurzschritt 79 / 126 / 191 / 274 / 343 / 410 / 448. Alle 5 Läufe je Zelle erreichen das Optimum. Auf 100 Zufalls- (12 × 12) und 100 Mischinstanzen (10 × 10, Seeds 0–99) erreichen alle vier Verfahren immer das Optimum; Mehrotra im Median 7 (höchstens 10 bzw. 11) Iterationen, Langschritt 16 bzw. 15, Affine Scaling 15 bzw. 14, Kurzschritt 236 bzw. 192 |
| **Kurzschritt gegen die Theorie** | Schranke ln(μ₀/ε) / −ln(1 − 0.4/√N): n = 4 / 8 / 16 / 32: gemessen 126 / 191 / 274 / 410 gegen 148 / 225 / 315 / 463, also **84 / 85 / 87 / 89 %** (die Theorie gilt bei zulässigem, zentriertem Start; hier gemessen mit Start ohne Zulässigkeit) |
| **Iterationen über ε** | Zufall 16 × 16, Mehrotra, ε = 10⁻² / 10⁻⁴ / 10⁻⁶ / 10⁻⁸ / 10⁻¹⁰ / 10⁻¹² / 10⁻¹³: **5 / 7 / 7 / 8 / 9 / 9 / 10** Iterationen, alle 5 Läufe erfolgreich; bei 10⁻¹⁴ nur noch 4 von 5 (unter Windows) |
| **Operationen gegen den Simplex** | Modell: Iteration 2m²N + m³/3 + 2m² je rechte Seite + 8mN + 10N, Pivot des dichten Tableaus 2(m+1)(Spalten+1). Verhältnis Innere Punkte / Simplex bei n = 2 / 4 / 8 / 16 / 24 / 32 / 40: **20.5 / 18.8 / 17.9 / 18.4 / 15.7 / 24.4 / 20.0** bei 1 / 2 / 4 / 9 / 15 / 14 / 21 Pivots. Kein Kreuzungspunkt |
| **Klee-Minty-Würfel** | Simplex (Dantzig) braucht 2^n − 1 Pivots (16383 bei n = 14); Mehrotra 5 / 9 / 12 / 13 / 16 / 17 Iterationen bei n = 2 / 6 / 8 / 10 / 12 / 14. Verhältnis der Operationen bei n = 2 / 4 / 6 / 7 / 8 / 10 / 14: **8.6 / 3.0 / 1.4 / 0.85 / 0.56 / 0.18 / 0.019**. **Kreuzungspunkt bei n = 7** (Ellipsoid, Stück 7: n = 13 bis 14) |
| **Startpunkt** | Zufall 16 × 16, Faktor auf den Mehrotra-Start 0.001 / 0.01 / 1 / 100 / 10⁸: Mehrotra erreicht **0 / 5 / 5 / 5 / 5 von 5** mit 25 / 8 / 13 / 16 Iterationen (bei 0.01 / 1 / 100 / 10⁸); Langschritt 0 / 3 / 5 von 5 bei 0.001 / 0.01 / 1. Zu kleiner Start: das Verfahren steht still und meldet einen falschen Verdacht auf Unbeschränktheit |
| **Schlechte Skalierung** | 5 feste 8 × 8-Instanzen, Spalten über 10^k gestreut (Optimalwert gleich), richtig gelöst bei k = 0 / 4 / 8 / 12 / 14 / 16: **Mehrotra 5 / 5 / 5 / 4 / 3 / 3 (Windows; unter Linux 5 / 5 / 5 / 4 / 5 / 5: ab 10¹² hängt es von der Plattform ab)**, Kurzschritt immer 5 (aber 165 bis 200 Iterationen), Langschritt 5 / 3 / 0 / 0 / 0 / 0, Affine Scaling 0 bis 1 bei 10¹² und 10¹⁶. Falsche Optima gab es nie, nur Stillstand. Mehrotra: Iterationen 6 / 17 / 24 bei k = 0 / 4 / 8 |
| **Kondition der Normalmatrix** | Zufall 16 × 16, ε = 10⁻⁸: größte Kondition je Instanz zwischen 10⁴ und 10⁷; bei ε = 10⁻¹⁴ steigt sie auf über 10¹² (in der Demo-Instanz auf 10¹⁶ und mehr), und die Iteration bricht ab |
| **Affine Scaling** | Auf 200 Zufallsinstanzen (8 × 8) nie gescheitert und nie mehr als das Dreifache von Mehrotra (Iterationen 10 bis 19 bei n = 2 bis 40). Das Fehlen einer Polynomialitäts-Garantie zeigte sich hier nicht; die bekannten schlechten Fälle sind konstruiert und nicht gebaut |
| **Unzulässig, unbeschränkt** | Unzulässig: das Lehrbuch mit Mindestmenge x1 ≥ 6 endet nach 2 Iterationen mit einem Farkas-Strahl y (Mᵀy ≤ 0, bᵀy > 0); unbeschränkt: nach 3 Iterationen ein Strahl z ≥ 0 (Mz = 0, cᵀz < 0). Beide sind nachgerechnet, also ein Beweis; ohne Strahl (Stillstand, Divergenz) meldet die Demo nur einen Verdacht |

## Vorab-Hypothesen

| Hypothese (vor der Messung) | Ergebnis |
|---|---|
| 20 bis 40 Iterationen, fast unabhängig von n | **Übertroffen:** Mehrotra braucht 4 bis 10 (bei n = 2 bis 40), Langschritt und Affine Scaling 10 bis 22; nur der Kurzschritt mit festem Faktor wächst deutlich (wie √N) |
| Mehrotra braucht etwa halb so viele Iterationen wie der Langschritt | **Bestätigt:** 4 gegen 11, 9 gegen 21 (Verhältnis 0.36 bis 0.43) |
| Kurzschritt liegt nahe an der Theorie-Schranke | **Bestätigt:** 84 bis 89 % der Schranke |
| Der Simplex gewinnt auf dichten Zufallsinstanzen | **Bestätigt:** das 16- bis 24-Fache im Modell, auf allen gemessenen Größen (n ≤ 40) |
| Innere Punkte gewinnen auf dem Klee-Minty-Würfel schon ab n ≈ 4 bis 6 | **Knapp verfehlt:** Kreuzungspunkt bei n = 7 (Verhältnis 1.39 bei n = 6, 0.85 bei n = 7) |
| Die Normalmatrix wird gegen Ende schlecht konditioniert und begrenzt die Genauigkeit | **Teils bestätigt:** bei ε = 10⁻⁸ bleibt sie unter 10⁷; erst nahe der Rundungsgrenze (ε = 10⁻¹⁴) steigt sie auf 10¹⁵ und mehr, und dort bricht das Verfahren ab |
| Innere Punkte bleiben bei schlechter Skalierung stabiler als Ellipsoid und Simplex | **Teils bestätigt:** Mehrotra und der Kurzschritt lösen bis 10¹⁶ fast alles richtig (ohne falsche Optima), Langschritt und Affine Scaling stehen ab 10⁴ bis 10⁸ still; die Iterationen von Mehrotra wachsen von 6 auf 17 bis 29 |
| Affine Scaling versagt auf zufälligen Instanzen manchmal | **Widerlegt:** 0 Ausreißer in 300 Instanzen (die bekannten Gegenbeispiele sind konstruiert) |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Der Pfad** (bei zwei Diensten die zulässige Menge, der Zentralpfad als Kurve der Zentren für fallendes μ, die Iterierten des Verfahrens mit Slider, der Optimalpunkt des Simplex; sonst μ, Lücke und Residuen über die Iterationen (log-y) und die Schrittlängen mit σ) → **Iterationen** (Tabelle aller vier Verfahren auf dieser Instanz; auf Abruf Iterationen über n und Kurzschritt gegen die Theorie) → **Gegen Simplex und Ellipsoid** (Operationen dieses Falls, auf Abruf über n und der Klee-Minty-Würfel mit Kreuzungspunkt) → **Grenzen** (Kondition der Normalmatrix; auf Abruf Genauigkeit verschärfen, Startpunkt variieren, schlechte Skalierung testen).
2. **Regler:** Instanz (Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung, Klee-Minty-Würfel, Unzulässig, Unbeschränkt), Größe (bis 40), Verfahren, Genauigkeit ε (10⁻² bis 10⁻¹⁴), Startpunkt (Faktor 0.001 bis 10⁸ auf den Mehrotra-Start), Skalierung der Spalten (bis 10¹⁶).
3. **Ergebnis:** "Optimum" bei relativer Lücke und Residuen unter ε; "unzulässig" bzw. "unbeschränkt" nur mit nachgerechnetem Strahl; "Verdacht" bei Stillstand, Divergenz oder Iterationsgrenze ohne Strahl; "numerisch gescheitert", wenn die Iterierten nicht mehr strikt positiv bleiben.

Presets (12): Lehrbuch: der Weg durch das Innere, Zentrum: wenige Iterationen, teure Schritte, Iterationen fast unabhängig von n, Vier Verfahren im Vergleich, Kurzschritt gegen die Theorie, Klee-Minty-Würfel: hier gewinnt das Innere, Zufall: der Simplex bleibt vorn, Startpunkt zu klein, Genauigkeitsgrenze, Schlechte Skalierung, Unzulässig: Farkas-Strahl, Unbeschränkt: Strahl.

## Modell und Verfahren

- **Instanz** (`ipm_scenario.py`): die Auslastungsplanung der Vorgängerstücke (Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung, Unzulässig, Unbeschränkt) und der Klee-Minty-Würfel in Chvátals Form (n ≤ 14); `column_scaled` rechnet die Spalten mit Faktoren zwischen 1 und 10^k um. Gleichungen sind hier kein Problem (anders als beim Ellipsoid).
- **Standardform** (`ipm_ipm.standard_form`): Schlupf je ≤-, Überschuss je ≥-Zeile, Minimierung von −c; abhängige Zeilen werden nach Zeilen- und Spaltenskalierung erkannt und entfernt, wenn ihre rechte Seite verträglich ist (sonst ist die Instanz unzulässig).
- **Newton-Schritt** (`ipm_ipm.newton_direction`): Normalgleichungen M (S⁻¹Z) Mᵀ Δy = r_p + M(S⁻¹Z r_d − S⁻¹ r_c) per Cholesky (dicht); danach Δs und Δz. Start nach Mehrotra (Kleinste-Quadrate-Lösung, verschoben und zentriert), Start ohne Zulässigkeit: die Residuen fallen mit den Schritten.
- **Vier Verfahren:** *Affine Scaling* (σ = 0, Schrittlänge 0.9 der Maximallänge), *Kurzschritt* (fester Faktor σ = 1 − 0.4/√N, Schrittlänge 0.9), *Langschritt* (σ = 0.1, Schrittlänge bis in die Umgebung z_j s_j ≥ 10⁻³ μ zurückgenommen), *Mehrotra* (Probeschritt, σ = (μ_aff/μ)³, Korrektor Δz_aff Δs_aff, Schrittlänge 0.9995 der Maximallänge). Der Kurzschritt der Theorie setzt einen zulässigen, zentrierten Start voraus; hier ist er die Variante mit festem Faktor.
- **Zertifikate:** ein y mit Mᵀy ≤ 0 und bᵀy > 0 (Unzulässigkeit) bzw. ein z ≥ 0 mit Mz = 0 und cᵀz < 0 (Unbeschränktheit), geprüft am normierten Iterierten in zeilen- und spaltenskalierten Größen (unabhängig von der Skalierung der Daten).
- **Zentralpfad** (`ipm_ipm.central_point`): für fallendes μ Newton-Zentrierung mit festem Ziel; nur für die Zeichnung bei zwei Diensten.
- **Auswertung** (`ipm_evaluation.py`): Einzellauf gegen den Simplex, Größen-, Genauigkeits- und Startpunktkurven, Kurzschritt gegen Theorie, Klee-Minty-Würfel, Skalierung, Verfahrenstabelle.

## Was nicht funktioniert hat / Grenzen

- **Meine erste Version hielt große Skalierung für Unzulässigkeit.** Der Rangtest der Zeilen und die Strahl-Zertifikate waren zunächst absolut gerechnet; bei Spalten über 10¹⁰ meldeten sie fälschlich "unzulässig" (mit Zertifikat!). Jetzt laufen beide in zeilen- und spaltenskalierten Größen; falsche Optima oder falsche Beweise gab es danach in keinem Lauf, nur Stillstand (Verdacht).
- **Zu kleine Startpunkte** lassen das Verfahren stillstehen; die Demo meldet dann einen Verdacht auf Unbeschränktheit, obwohl die Instanz beschränkt ist. Der Verdacht ist ausdrücklich kein Beweis; belegt sind nur nachgerechnete Strahlen.
- **Die Iterationen von Mehrotra wachsen bei schlechter Skalierung** (6 auf 17 bis 29), und bei 10¹² bleibt je nach Plattform ein Lauf von fünf hängen oder keiner; die Schrittlängen-Regel des Langschritts (Umgebung N₋∞) ist das empfindlichste Stück.
- **Kein Kreuzungspunkt auf Zufallsinstanzen.** Bis n = 40 (Regler) liegt das Verfahren im Modell beim 16- bis 24-Fachen des dichten Simplex; die Stärke der Inneren Punkte (dünne Faktorisierung großer Matrizen, Präsolve, dichte Spalten) bildet die Demo nicht ab. Das Operationsmodell zählt keine Speicherzugriffe.
- **Nur die Grundformen:** kein Karmarkar-Original (projektive Transformation), kein Dikin-Affine-Scaling im primalen Raum (die Demo hat das primal-duale mit σ = 0), keine homogene selbstduale Einbettung (Unzulässigkeit wird über Strahlen der Iterierten erkannt, nicht über die Einbettung), keine höheren Ordnungen, keine Nachbehandlung (Crossover, Stück 9 der Reihe: aus dem Inneren eine Ecke und eine Basis zurückgewinnen).
- **Der Startpunkt der Theorie fehlt.** Der Kurzschritt ist die Variante mit festem Faktor bei Start ohne Zulässigkeit; die Beweisschranke gilt nur für zulässigen, zentrierten Start (dort exakt getestet), gemessen liegt er bei 84 bis 89 % davon.
- **Die Bruchstelle der Genauigkeit hängt von der Plattform ab** (ε = 10⁻¹⁴: unter Windows 4 von 5 Läufen, sonst kann es abweichen); bis ε = 10⁻¹³ gelingt das Zertifikat auf allen gemessenen Instanzen.
- **Synthetische, kleine Instanzen** (bis 40 × 40, Würfel bis 14); Skalierungstest auf fünf Instanzen.

## Verifikation

- `tests/test_algorithm.py`: **Newton-Richtung gegen das volle KKT-System**; Standardform; **Optimum gegen HiGHS und Simplex auf über 300 Läufen** für alle vier Verfahren; Dualwerte; strikte Innerlichkeit und Umgebung; **Kurzschritt-Reduktion exakt bei zentriertem Start**; Kurzschritt gegen die Schranke; Mehrotra-σ gegen unabhängige Neuberechnung; Affine Scaling; **Farkas- und Unbeschränktheits-Strahlen** (Fixtures, 60 unzulässige, 40 unbeschränkte konstruierte Instanzen, jedes Verfahren); redundante Zeilen (verträglich entfernt, widersprüchlich unzulässig); Startpunkt-Extreme; Buchführung des Operationsmodells; jeder Zweig.
- `tests/test_scenario.py` (auch der Würfel gegen HiGHS), `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen; Iterationszahlen mit kleinem Band, weil Gleitkomma-Rundung sie plattformabhängig verschieben kann; die Genauigkeitsgrenze nur mit Sicherheitsabstand), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und jedes Verfahren, Iterations-Regler, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy, pandas, plotly und streamlit.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Karmarkar, N. (1984). *A new polynomial-time algorithm for linear programming.* Combinatorica 4(4), 373–395.
- Mehrotra, S. (1992). *On the implementation of a primal-dual interior point method.* SIAM Journal on Optimization 2(4), 575–601.
- Wright, S. J. (1997). *Primal-Dual Interior-Point Methods.* SIAM.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
