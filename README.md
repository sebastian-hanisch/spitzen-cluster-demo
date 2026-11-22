# Spitzen-Cluster: der Extremalindex der Wartezeit – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-spitzen-cluster-demo.streamlit.app/)**

Demo der Linie **„Extremwerte erklärt“** in der „Konzepte“-Reihe der Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Bereich Statistik, Feld „Seltenes und Extremes“). Die Demos davor in der Linie (von „Das Größte von n“ bis „Pegel statt Index“) setzen voraus, dass die Beobachtungen **unabhängig** sind; diese Demo hebt genau diese Annahme auf (Art der Kante: `fix`, „unabhängig → abhängig“) und besitzt in der Linie die Frage „Abhängigkeit“. **Stationär, aber abhängig:** die Verteilung ändert sich nicht, die Werte hängen aneinander; „stationär“ hebt erst die Demo „Trend“ auf. Vehikel ist die Wartezeit einer **M/M/1-Schlange** (Dienstrate 1, Ankunftsrate ρ = Last, unendlicher Warteraum) und, als zweites exaktes Orakel, der **max-AR(1)-Prozess** mit Fréchet-Rand. Das Modell ist **ausgedacht**: gezeigt wird, was in ihm geschieht, nicht, was das für einen Betrieb heißt.

## Warum dieses Problem

Hohe Wartezeiten kommen im Gleichschritt. Der **Extremalindex θ ∈ (0, 1]** (Kehrwert der mittleren Clustergröße, gezählt in Überschreitungen je Cluster, im Grenzwert hoher Schwellen; nicht der Extremwertindex ξ; θ = 1 heißt nicht „unabhängig“) macht das messbar. **Kernfrage:** Wie stark häufen sich die Spitzen, was kostet es, sie wie unabhängig zu behandeln, und wie oft hält ein 95-%-Intervall für den Pegel sein Versprechen? 3.000 Kunden sind keine Stichprobe vom Umfang 3.000: unter 150 Überschreitungen stecken bei Last 80 % rechnerisch nur k·θ = 6 unabhängige Cluster.

## Befunde und Korrekturen (gegenüber Plan und Vormessung)

Plan und Vormessungen (V16, V2, V18, Nachzug) rechneten mit `numpy.default_rng` und scipy; die Demo misst neu mit SplitMix64 (`clu_rng.py`, als Zähler-Strom in `clu_queue.py`), anderem Zufalls-Seed und mehr Läufen (Abdeckung R = 2.000 je Zelle statt 500). Zufallsanteile stehen als Standardfehler.

**Korrektur des Delta-Intervalls (2026-11-20, nach unabhängiger Prüfung):** Die erste Fassung dieser Demo rechnete die Kovarianz von σ̂ und ξ̂ der GPD mit dem Nebenglied **+**σ(1+ξ)/k; richtig ist **−**σ(1+ξ)/k. Das Intervall war dadurch viel zu breit: an unabhängigen Daten deckte es bei 12.000 Kunden und 600 Überschreitungen über 99 % statt rund 93 % (Test), und die gemeldete Abdeckung bei Abhängigkeit war viel zu hoch. Belegt ist das Vorzeichen jetzt durch eine Simulation (Kovarianz von σ̂ und ξ̂ bei ξ = 0, σ = 1, k = 600 Überschreitungen: −0,0018 gegen −0,0017 der Matrix) und durch die Abdeckung an unabhängigen Beobachtungen (Test `test_oracle_gpd`); die Complex-Step-Fassung des Orakels teilt die Matrix mit dem Code und hätte den Fehler nicht gefunden. Alle abhängigen Zahlen sind neu gemessen (Läufe, Seeds und Rechenwege unverändert, nur die Matrix); die Zahlen vor der Korrektur stehen zum Vergleich in der Tabelle und sind im Test `test_oracle_gpd` mit dem früheren Vorzeichen für die Hauptzellen reproduziert. Die Aussage „mit exaktem θ mindestens 93 % in 13 von 36 Zellen“ der ersten Fassung gilt nicht mehr (jetzt: in keiner Zelle).

Jede verschobene Zahl:

| Größe | Plan / Vormessung | Neu gemessen | Grund oder Lesart |
|---|---|---|---|
| θ(u) bei Last 0,9 und F̄ = 10⁻⁶ (Kern, h = 0,1) | 0,0103 (+3,5 %, „Randeffekt, Schwelle nahe der Gittergrenze“) | **0,0100** | die Vormessung ließ nur 33 Abfertigungsdauern Rand über der Schwelle; mit 60 ist θ(u) stabil. Bei nur 6 Rand verfälscht der letzte Gitterzustand θ(u) bei F̄ = 10⁻³ um 32 % (Last 0,8) bis 99 % (Last 0,9) (Test). |
| θ(u) bei Last 0,5 und F̄ = 10⁻⁶ | 0,2501 | 0,2499 | Gitterfehler (h = 0,1), im Rahmen |
| Verschiebung des Tagesmaximums, Last 80 %, n = 256, Median | 14,4 (28,4 − 14,0 aus gerundeten Zahlen) | **14,5** | aus den ungerundeten Quantilen; Quantil 0,99 und 0,999: 15,3 und 15,4, also 43 bis 46 min bei 3 min Abfertigungsdauer wie im Plan |
| Populations-ξ der besten GEV (Kolmogorov), Last 80 %, n = 64 / 256 / 1.024 / 4.096 | +0,155 / +0,086 / +0,043 / +0,018 | +0,159 / +0,087 / +0,043 / +0,018 | anderes Raster (0,5 statt 40 Punkte zwischen 1 % und 99,9 %); Kolmogorov-Abstand 0,0050 / 0,0028 / 0,0016 / 0,0007 gegen 0,0055 / 0,0029 / 0,0016 / 0,0007 |
| Abdeckung Last 50 %, 3.000 Kunden, 5 % über der Schwelle, p = 10⁻³ (unabhängig / θ̂ / θ exakt) | 70,4 / 93,2 / 96,4 % (R = 500) | **35,4 / 59,1 / 61,7 %** (R = 2.000, Standardfehler 1,1 / 1,1 / 1,1 Punkte); erste Fassung dieser Demo mit dem falschen Vorzeichen 73,9 / 94,8 / 97,4 % | Vorzeichenfehler der Kovarianz (siehe oben); die Vormessung rechnete mit einer Matrix derselben Gestalt (nicht nachgeprüft), ihre Zahlen werden nicht mehr zitiert |
| dasselbe bei Last 80 % | 16,8 / 72,0 / 88,2 % | **4,9 / 22,9 / 25,6 %** (Standardfehler 0,5 / 0,9 / 1,0); erste Fassung 17,9 / 70,2 / 86,9 % | dieselbe Ursache |
| Alle 36 Zellen der Abdeckung, „unabhängig“ / θ̂ / θ exakt, kleinster bis größter Wert | 4,2 bis 76,9 % / 15,4 bis 97,0 % / 37,3 bis 98,5 % (erste Fassung, falsches Vorzeichen) | **1,0 bis 45,4 % / 3,3 bis 69,9 % / 10,2 bis 74,2 %** | mit exaktem θ erreichten in der ersten Fassung 13 von 36 Zellen mindestens 93 %; jetzt keine |
| θ̂ (Intervall-Schätzer, Median) und Verzerrung, Last 50 % / 80 % | 0,295 / 0,064; Verzerrung −0,106 (77 % der Läufe unter dem wahren Pegel) / −0,316 (89 % der Läufe) | 0,297 / 0,063; −0,085 (73 % der Läufe) / −0,323 (90 % der Läufe) | im Rahmen |
| Block gegen Schwelle am max-AR(1), Schwelle ohne θ, k = 50, φ = 0 / 0,5 / 0,8 | 95,8 / 80,3 / 50,2 % | **89,0 / 62,4 / 32,6 %** (R = 1.500, Standardfehler 0,8 bis 1,3 Punkte); erste Fassung 95,3 / 78,3 / 49,1 % | Vorzeichenfehler der Kovarianz; auch bei φ = 0 (unabhängige Fréchet-Daten, k = 50) deckt das korrigierte Intervall nur 89,0 % |
| dasselbe mit exaktem θ | 95,8 / 90,8 / 81,1 % | **89,0 / 75,6 / 60,5 %**; erste Fassung 95,3 / 89,5 / 82,0 % | dieselbe Ursache; mit k = 125: 92,2 / 81,9 / 72,1 % (erste Fassung 95,9 / 92,5 / 87,4 %) |
| Block-GEV, ln-Skala, φ = 0 | 95,3 % | **93,1 %** | die Block-GEV ist hier ein eigener Anpasser (Nelder-Mead, zwei Startwerte) mit beobachteter Information, vom Vorzeichenfehler nicht betroffen; 3,4 Standardfehler unter der Vormessung; die ln-Skala liegt bei φ = 0,5 und 0,8 bei 93,8 und 91,6 %. Die Lesart „der Block verliert durch die Abhängigkeit nichts Messbares“ der ersten Fassung ist eingeschränkt: er verliert weniger als der Peaks-over-Threshold-Ansatz (POT), aber messbar (siehe Befunde) |
| Läufe der Abdeckungsstudie live | „R = 5 bei n = 3.000“ wegen der Dezimal-Logarithmen der Vorlage | **R = 200 bei n = 12.000 in 0,9 bis 6,8 s** je nach Last und Anteil über der Schwelle (Windows, ein Thread, Maschine unter fremder Last) | der Strom liegt als numpy-Zähler vor (bitgleiche Gleichverteilte, Exponentialzahlen mit `numpy.log` auf 1e-12 gleich der Vorlage, Test); Streamlit Cloud nicht gemessen |
| Experiment 1: Lauflänge | „bis 20.000 nur aus eingefrorenen Reihen“ | bis 20.000 live | aus demselben Grund |

**Nicht untersucht und nicht verschoben:** Die V2-Quantile des Tagesmaximums aus einer langen Folge (14,0 / 36,1 / 49,0) werden nicht zitiert; die exakten Werte sind 14,0 / 34,3 / 45,8 (Test). Der Pegel der GPD-Anpassung liegt bei Abhängigkeit im Mittel unter dem wahren (**negative** Verzerrung: in allen 36 Zellen liegt die mittlere Verzerrung von ln unter null, 62 bis 99 % der Läufe unter dem wahren Pegel); diese Zahlen hängen nicht am Intervall und blieben bei der Korrektur gleich. Ob und warum das mit der Literatur übereinstimmt, ist nicht geklärt (siehe Quellen).

## Modell

Dienstrate 1 (die Abfertigungsdauer ist die Zeiteinheit), Ankunftsrate ρ (Last), Wartezeit W_{n+1} = max(0, W_n + S_n − A_{n+1}) mit S ~ Exp(1), A ~ Exp(ρ), γ = 1 − ρ; **stationär gestartet** (W_1 = 0 mit Wahrscheinlichkeit 1 − ρ, sonst Exp(γ)), ohne Einschwingzeit. Exakt: P(W > x) = ρ e^(−γx), die Überschreitungen jeder Schwelle sind exponentiell (ξ = 0), der Pegel zur Überschreitungswahrscheinlichkeit p **je Kunde** ist x_p = ln(ρ/p)/γ. „Pegel“ ist ein Wiederkehrwert, keine Frist. **Tagesmaximum** = größte Wartezeit unter n = 256 aufeinanderfolgenden Kunden (Annahme dieser Seite). **max-AR(1):** X_t = max(φ X_{t−1}, (1−φ) Z_t), Z ~ Fréchet(1), Rand Fréchet(1), θ = 1 − φ; das Maximum über b aufeinanderfolgende Werte ist exakt Fréchet mit der Skala s_b = 1 + (b−1)(1−φ). Formeln im 📐-Expander der App. Vorzeichen: ξ > 0 schwerer Rand; `scipy.stats.genextreme` hat c = −ξ, `genpareto` c = ξ (nur in Tests).

## Methodik

1. **Eine Reihe** (Experiment 1): Schwelle mit exaktem Randanteil, Cluster nach der Runs-Regel, drei Schätzer von θ (Runs mit r, Blöcke mit b, Intervalle nach Ferro/Segers) neben dem exakten θ und dem Kern θ(u); dazu die eingefrorene Tabelle der Mediane über 500 Läufe.
2. **Tagesmaximum** (Experiment 2): exakte Verteilung π Q^(n−1) 1 aus dem Gitterkern (h = 0,1, eingefroren) gegen die Gumbel-Näherung mit θ = (1−ρ)² und mit θ = 1.
3. **Intervall** (Experiment 3): GPD-Maximum-Likelihood an die Überschreitungen einer empirischen Schwelle, Delta-Intervall für ln x_p, drei Varianzen (durch 1, durch θ̂, durch θ exakt geteilt); 36 eingefrorene Zellen mit R = 2.000 Läufen, dazu Live-Läufe.
4. **Blocklänge** (Experiment 4): beste GEV an die exakte Verteilung des Blockmaximums (Kolmogorov-Abstand und Likelihood), eingefroren.
5. **Block gegen Schwelle bei Abhängigkeit** (Experiment 5): max-AR(1), 50 Blöcke zu 50 Beobachtungen, Ziel der 100-Block-Wert; die GPD beim **θ-bereinigten Niveau** p* (der 100-Block-Wert hat bei θ < 1 je Beobachtung die Überschreitungswahrscheinlichkeit p* = 1 − exp(−1/x*) statt 1 − 0,99^(1/50)).

Laufzeit der App rechnet mit numpy und der Standardbibliothek; scipy steht nur in `requirements-dev.txt`, in Tests und in den Werkzeugen.

## Befunde (gemessen, keine Behauptungen)

„exakt“ heißt Bruch, Ganzzahl oder geschlossene Formel; „Kern“ heißt Gitter (h = 0,1, kein Zufall); „Messung“ heißt eingefrorene Läufe (Zähleinheit in der Spalte), im Test gegen die Datei gelesen und als Band neu gerechnet. Gilt für dieses Modell (unendlicher Warteraum, stationär, ξ = 0), Last bis 0,9.

| Frage | Befund | Art | Test |
|---|---|---|---|
| θ der M/M/1-Wartezeit | **θ = (1−ρ)² = 1/4, 1/25, 1/100** bei Last 50 / 80 / 90 %: im Mittel **4, 25, 100 Überschreitungen je Cluster** (Grenzwert hoher Schwellen); ein Cluster kann Lücken enthalten und überspannt dann mehr Kunden als Überschreitungen (nicht gemessen) | exakt (eigene Herleitung, zwei Rechenwege) | `test_handbeispiele` H 1, `test_oracle_theta` |
| Diskretes Gegenstück (andere Größe: Sprungkette) | θ = 1/6; P(W_1..W_40 ≤ 5 \| W_0 > 5) = 0,159187 (exakter Bruch, unter 1/6) | exakt | H 2 |
| θ(u) aus dem Kern bei F̄ = 10⁻² / 10⁻³ / 10⁻⁴ / 10⁻⁶ | Last 50 %: 0,2612 / 0,2516 / 0,2501 / 0,2499; Last 80 %: 0,0427 / 0,0404 / 0,0400 / 0,0400; Last 90 %: 0,0107 / 0,0101 / 0,0100 / 0,0100. Bei F̄ = 10⁻³ liegt θ(u) rund 1 % über dem Grenzwert, bei F̄ = 10⁻² 4 bis 8 %; das Gitter h = 0,2 weicht um höchstens 0,3 % ab | Kern | `test_claims`, `test_oracle_theta` |
| Tagesmaximum, 256 Kunden, Quantile 0,1 / 0,5 / 0,99 / 0,999 (Abfertigungsdauern) | Last 50 %: 5,5 / 7,8 / 16,2 / 20,8; Last 80 %: 9,2 / 14,0 / 34,3 / 45,8; Last 90 %: 11,9 / 20,0 / 59,5 / 82,6 | Kern, gegen Simulation im Band | `test_claims`, `test_oracle_theta` |
| Dasselbe „unabhängig gerechnet“ (Gumbel, θ = 1) | Last 80 %: 22,4 / 28,4 / 49,6 / 61,1; Verschiebung 13,3 / 14,5 / 15,3 / 15,4 Abfertigungsdauern, bei 3 min Abfertigung **43 bis 46 min** für Median bis Quantil 0,999; der Grenzwert ln(1/θ)/γ = 16,1 (48 min) gilt für unendlich viele Kunden | Kern und Formel | `test_claims` |
| Gumbel mit θ = (1−ρ)² | am oberen Rand brauchbar bei Last 80 % und 256 Kunden (33,5 gegen 34,3 und 45,1 gegen 45,8, also −2,2 und −1,6 %), bei Last 90 % weniger gut (−8,7 und −6,2 % gegen exakt, am 0,99- und 0,999-Quantil), im Körper falsch (0,1-Quantil 6,3 gegen 9,2; Last 90 %: Median 12,0 gegen 20,0, 0,1-Quantil 0,0 gegen 11,9). Cluster je Tag n·θ·ρ = 8,2 (Last 80 %) und 2,3 (Last 90 %) | Kern und Formel | `test_claims` |
| Populations-ξ der Blockmaxima, Last 80 %, 64 / 256 / 1.024 / 4.096 Kunden je Block | ξ = **+0,159 / +0,087 / +0,043 / +0,018** (beste GEV nach Kolmogorov-Abstand), +0,151 / +0,083 / +0,040 / +0,016 (Likelihood), obwohl der Grenzwert ξ = 0 ist (Populationswerte, nicht die Werte einer Anpassung an eine Stichprobe von Tagesmaxima; bei Last 50 % und 4.096 Kunden ist der Wert nach Kolmogorov mit +0,003 praktisch null); Gumbel-Anpassung: Abstand 0,0216 / 0,0120 / 0,0059 / 0,0025 gegen 0,0055 / 0,0029 / 0,0016 / 0,0007 der GEV; σ = 3,56 / 4,12 / 4,56 / 4,82 (Grenzwert 5,0). Last 50 %: +0,057 / +0,025 / +0,010 / +0,003 | Kern | `test_claims`, `test_oracle_gpd` |
| Abdeckung des 95-%-Intervalls, Last 50 %, 3.000 Kunden, 150 Überschreitungen (5 %), p = 10⁻³ je Kunde, R = 2.000 Läufe | „unabhängig“ **35,4 %**, mit θ̂ 59,1 %, mit exaktem θ 61,7 % (Standardfehler 1,1 / 1,1 / 1,1 Punkte); k·θ = 37,5 Cluster; θ̂ im Median 0,297 gegen 0,25 | Messung (Läufe) | `test_claims` |
| Dasselbe bei Last 80 % | **4,9 %**, 22,9 %, 25,6 % (Standardfehler 0,5 / 0,9 / 1,0); k·θ = 6 Cluster; θ̂ im Median 0,063 gegen 0,04; Pegel unter dem wahren in 90 % der Läufe, Verzerrung von ln −0,323; Breite des Intervalls ×2,0 (Last 50 %) und ×5,0 (Last 80 %) gleich 1/√θ (Konstruktion: die Varianz wird durch θ geteilt, kein Messbefund) | Messung (Läufe) | `test_claims` |
| Warum das Intervall mit exaktem θ trotzdem zu eng ist | Die Streuung von ln des Pegels (Standardabweichung 0,194 bei Last 50 %, 0,240 bei Last 80 %) ist das 2,0- und 2,7-Fache der Standardabweichung, die das Intervall mit exaktem θ unterstellt (Median der halben Breite durch 1,96: 0,099 und 0,087); in allen 36 Zellen mindestens das 1,6-Fache. Bei Last 50 %, 12.000 Kunden und 600 Überschreitungen streut ln des Pegels etwa das 3,3-Fache der Streuung unabhängiger Daten (Test, Band 2,3 bis 4,2), das Teilen der Varianz durch θ = 1/4 unterstellt das Doppelte. Die Überschreitungen eines Clusters hängen in ihrer Größe zusammen, k Überschreitungen sind also weniger als k·θ unabhängige Beobachtungen der Überschussverteilung (Deutung, nur an dieser Zerlegung gestützt) | Messung (Läufe) und Test | `test_claims`, `test_oracle_gpd` |
| Dasselbe Intervall an unabhängigen Beobachtungen (θ = 1) aus der Randverteilung der Wartezeit, Last 50 %, p = 10⁻³ | bei 3.000 Kunden und 150 Überschreitungen deckt es rund 89 % (nicht 95 %: Näherung des Delta-Intervalls bei wenigen Überschreitungen), bei 12.000 Kunden und 600 Überschreitungen rund 93 %; Streuung des Pegels und mittlere Intervallbreite stimmen im Test auf wenige Prozent überein (Band ±15 %) | Messung im Test (R = 500 und 300) | `test_oracle_gpd` |
| Alle 36 Zellen (Last 50 / 80 %, n = 1.000 / 3.000 / 12.000, Anteil 2 / 5 / 10 %, p = 10⁻³ / 10⁻⁴) | „unabhängig“ 1,0 bis 45,4 %, mit θ̂ 3,3 bis 69,9 %, mit exaktem θ 10,2 bis 74,2 %: in keiner Zelle erreicht eine der drei Varianten 93 %; die Abdeckung mit exaktem θ steigt mit der Zahl der Cluster nur langsam (bei Last 50 %: 61,7 % bei 37,5 Clustern in der Hauptzelle, 71,7 % bei 12.000 Kunden und 300 Clustern k·θ); geschätztes θ im Median über dem exakten in 36 von 36 Zellen, und mit θ̂ deckt das Intervall in 36 von 36 Zellen weniger als mit exaktem θ; bei p = 10⁻⁴ deckt die Korrektur mit exaktem θ in 13 von 18 Paaren weniger als bei p = 10⁻³; in allen 36 Zellen Verzerrung von ln unter null, 62 bis 99 % der Läufe unter dem wahren Pegel | Messung (Läufe) | `test_claims` |
| Schätzer von θ, max-AR(1), 5.000 Beobachtungen, 5 % über der Schwelle, 500 Läufe, Median | φ = 0 (unabhängig, θ = 1): Runs mit r = 10 **0,597**, Blöcke (b = 50) 1,00, Intervalle 1,00 (die dritte Stelle ist über Zufalls-Seeds nur auf etwa ±0,01 stabil); φ = 0,5 (θ = 0,5): 0,383 / 0,510 / 0,499; φ = 0,8 (θ = 0,2): 0,179 / 0,220 / 0,206. Bei 1 % über der Schwelle und φ = 0: Runs 0,912 | Messung (Läufe) | `test_claims`, `test_oracle_theta` |
| Schätzer an der M/M/1-Wartezeit, 100.000 Kunden, 40 Läufe, Schwelle bei 1 % | Last 50 %: Runs 0,26, Blöcke 0,29, Intervalle 0,27 (θ(u) 0,26); Last 80 %: 0,05 / 0,07 / 0,04 (θ(u) 0,04); Last 90 %: 0,02 / 0,04 / 0,01 (θ(u) 0,01): Runs mit festem r = 10 liegt bei hoher Last weit über θ(u), weil Lücken im Cluster länger als 10 Kunden sind (die dritte Stelle der Mediane streut bei 40 Läufen über Zufalls-Seeds, darum zwei Stellen) | Messung (Läufe) | `test_claims` |
| Block gegen Schwelle am max-AR(1), 50 Blöcke zu 50, k = 50 (125), R = 1.500, φ = 0 / 0,5 / 0,8 | Block-GEV auf der Pegelskala 88,9 / 86,1 / 84,3 %, ln-Skala 93,1 / 93,8 / 91,6 %; Schwelle (θ-bereinigtes Niveau) ohne θ 89,0 / 62,4 / 32,6 % (125: 92,2 / 68,5 / 38,7 %), mit exaktem θ 89,0 / 75,6 / 60,5 % (125: 92,2 / 81,9 / 72,1 %); Cluster k·θ = 50 / 25 / 10 (125: 125 / 62,5 / 25) | Messung (Läufe) | `test_claims` |
| Konstruktion des Ziels | 100-Block-Wert x\* = s_b/(−ln 0,99); p\* = 2,01·10⁻⁴ / 3,94·10⁻⁴ / 9,30·10⁻⁴ (φ = 0 / 0,5 / 0,8) gegen 1 − 0,99^(1/50) = 2,01·10⁻⁴: bei θ < 1 das 1,96- und 4,63-Fache; die erste Fassung (Niveau 1 − 0,99^(1/50), exaktes θ) deckt 83,3 und 87,7 % bei φ = 0,8 und 0,5 (bei k = 50; die bereinigte Fassung 60,5 und 75,6 %) und trifft das Ziel bei k = 50 im Median, wahrscheinlich weil sich zwei Abweichungen aufheben (nicht untersucht) | exakt und Messung | H 7, `test_claims` |
| Live-Rechenzeit (diese Maschine, Windows, ein Thread) | Reihe mit 20.000 Kunden samt θ(u) aus dem Kern unter 0,2 s; 200 Läufe zu 12.000 Kunden 0,9 bis 6,8 s je nach Last und Anteil über der Schwelle; ganzer AppTest-Lauf der Seite 1,0 bis 1,3 s (frühere Messung, nicht wiederholt) | Messung | nicht im Test (Zeit hängt an der Maschine) |

**Ehrlicher Negativbefund:** Auch die Korrektur mit dem richtigen θ reicht nicht: bei Last 80 % und 150 Überschreitungen (6 Cluster) deckt sie 25,6 %, bei Last 50 % und 150 Überschreitungen (37,5 Cluster) 61,7 %, nicht 95 %, und von 37,5 auf 300 Cluster (Last 50 %) steigt die Abdeckung mit exaktem θ nur von 61,7 auf 71,7 % (siehe Befunde: die Streuung des Pegels ist größer, als das Teilen der Varianz durch θ unterstellt). Das geschätzte θ liegt im Median über dem exakten, das Intervall wird damit noch enger. Der POT-Ansatz ohne θ fällt am max-AR(1) bei φ = 0,8 auf 32,6 % (k = 50), mit exaktem θ auf 60,5 %; der Block-Maxima-Ansatz verliert durch die Abhängigkeit **weniger** als der POT-Ansatz: auf der ln-Skala 93,1 → 91,6 %, auf der Pegelskala 88,9 → 84,3 % (φ = 0 → 0,8; bei Blöcken von 50 Beobachtungen gegen 1/θ = 5; andere Blocklängen nicht gemessen). Beim POT-Ansatz ist offen, wie viel am Delta-Intervall und wie viel an der GPD-Anpassung bei Clustern liegt; die Demo trennt beides nicht.

## Ehrliche Grenzen

* **Ausgedachtes Modell.** Es gibt in den gelesenen Quellen keine reale Wartezeitreihe eines Gates oder Terminals, an der Spitzen-Quantile geschätzt und später geprüft wurden. Was das für einen Betrieb heißt, wird nicht behauptet; auch die 3 min je Abfertigung sind nur eine Umrechnung.
* **θ = (1−ρ)² gilt als Grenzwert hoher Schwellen.** Bei F̄ = 10⁻³ liegt θ(u) rund 1 % darüber, bei F̄ = 10⁻² 4 bis 8 % (Kern). 1/θ ist der Kehrwert dieses Grenzwerts und zählt Überschreitungen je Cluster, nicht Kunden: ein Cluster kann Lücken enthalten und mehr Kunden überspannen.
* **Der Kern ist ein Gitter** (h = 0,1, bei θ(u) in der App h = 0,2), kein exakter Wert im strengen Sinn; er ist gegen die geschlossene Randverteilung (n = 1), eine zweite Gitterweite, die Simulation der Definition und gegen die Zerlegungsformel geprüft. Das 0,1-Quantil des Tagesmaximums hat einen Gitterfehler bis ±0,1.
* **Die Gumbel-Näherung mit θ braucht viele Cluster je Tag.** Als Faustregel passen Werte von n·θ·ρ ab etwa 8 (Abweichung am 0,99-Quantil −2,2 % bei 8,2, −8,7 % bei 2,3); eine bewiesene Bedingung ist das nicht, und n·θ·ρ zählt die Cluster oberhalb des Niveaus null, nicht oberhalb eines hohen Niveaus. Im Körper ist die Näherung bei beiden Werten falsch. Bei sehr wenigen Clustern kann das Quantil null oder negativ werden.
* **Das Intervall ist eine Konstruktion:** Skala (ln) und der Anteil der Varianz, der durch θ geteilt wird, sind Wahlen; sie sind hier nicht variiert (die Zahlen des Nachzugs der Vormessung galten für das Intervall mit falschem Vorzeichen und werden nicht mehr zitiert). Gemessen ist, dass das Teilen durch θ bei M/M/1-Abhängigkeit zu wenig verbreitert (siehe Befunde). Die Formel stammt aus dem Gedächtnis (Smith 1987, Stufe G); das Vorzeichen des Nebenglieds war falsch und ist jetzt an einer Simulation geprüft, die übrigen Glieder an der Streuung und Abdeckung unabhängiger Daten, nicht an der Quelle.
* **Der Intervall-Schätzer von θ** (Ferro und Segers 2003) ist aus dem Gedächtnis (Stufe G), an exakten Werten des max-AR(1) und an einem Handbeispiel geprüft.
* **Warum der GPD-Pegel bei Clustern im Median tiefer liegt** (0,55 / 0,42 des Ziels bei φ = 0,5 / 0,8 in Experiment 5, 0,82 bei φ = 0) ist nicht untersucht.
* **Zufallsanteile** sind Bänder mit Standardfehler, nie ein Test auf das letzte Bit (Linux-CI); Docker-/Linux-Prüfung (`demo_linux_check`) und Streamlit Cloud sind nicht gelaufen.

## 🚧 Bewusst nicht enthalten / nicht gemessen

* **Hill-Schätzer und Schwellenwahl unter serieller Abhängigkeit** sowie Block gegen Schwelle bei anderen Blocklängen unter Abhängigkeit: nicht gemessen; die Seite zeigt nur die Wirkung auf den Pegel bei fester Schwelle und die Zahl der Cluster.
* **Pegel des Tagesmaximums** (statt je Kunde) mit Intervallen, Profil-Likelihood und Blockbootstrap am Vehikel A: nicht gemessen.
* **Last über 0,9, Reihen unter 1.000 Kunden** (außer in Experiment 2): nicht gemessen; die Schlange ist ab ρ = 1 nicht stationär (die App lehnt sie ab).
* **Endlicher Puffer** (Gate M/M/2/6): in der Vormessung ein deutlich größeres θ, hier nicht neu gemessen und nicht in der App.
* **Subexponentielle Bedienzeit** (nach Literatur der Stufe A ist θ dort null), **ξ > 0 zusammen mit Abhängigkeit**, **Gleichstände**: nicht gemessen.
* **Last 0,9 mit Schwelle bei 0,1 % und 100.000 Kunden** hat erwartet 100 Überschreitungen, aber nur rund einen Cluster je Lauf (n·F̄·θ = 1): die Schätzer sind dort in 12 von 40 Läufen nicht definiert. Die Zahl ist ein Zufallswert aus 40 Läufen und ändert sich mit dem Zufalls-Seed; ohne Cluster bleibt rund ein Drittel der Läufe (Tabelle der Mediane in Experiment 1).
* **Randabhängigkeit zweier Größen** (Kopula) ist keine zeitliche Abhängigkeit einer Folge: nicht hier.
* **Keine Aussage über Nutzen** in Logistik, Lager oder Hafen; das Deich-Beispiel der Lehrbücher (Meterangaben sind Annahmen des Buchs) steht nicht in der App.

## Quellen (Stufe der Lektüre: W = in der Quellenliste der Planung als gelesen geführt und hier nicht erneut gelesen, A = Zusammenfassung oder Kurzseite, G = aus dem Gedächtnis, sonst nicht gelesen)

* Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*, Satz 5.2 und Gleichung (5.4): Definition und Kehrwert-Deutung von θ; Stufe W, nur Gleichungen und Befunde in eigenen Worten.
* Embrechts, Klüppelberg, Mikosch (1997), *Modelling Extremal Events*, Definition 8.1.2: Stufe W.
* **Peshkova (2026)**, *Discrete Contin. Models Appl. Comput. Sci.* 34(1), 24–39: nennt laut Journal-Seite θ = (1−ρ)² für M/M/1 (Stufe A); das PDF liegt als Binärdatei vor, **der Wortlaut ist nicht gelesen**, Abschnitt und Gleichungsnummer sind nicht geprüft. Die Quelle bleibt auf Stufe A; **die Herleitung in der App und hier ist eigene Arbeit** mit zwei Rechenwegen (Zerlegungsformel in Brüchen, dominanter Eigenwert des abgetöteten Kerns).
* Hooghiemstra, Meester (1996), Rootzén (1988), Asmussen, Perry (1992): nur Suchtreffer oder Zusammenfassungen, nicht gelesen, nicht benutzt.
* Ferro, Segers (2003) (Intervall-Schätzer), Smith (1987) (erwartete Information der GPD; das Vorzeichen des Nebenglieds stand zuerst falsch, jetzt an einer Simulation geprüft), Davis, Resnick (1989) und Chernick, Hsing, McCormick (1991) (max-AR(1), θ = 1 − φ): Stufe G. arXiv 2403.16590 (Max-ARMA(1,0)): Stufe A.
* Belzile, Davison (arXiv 2007.10780): behandeln unabhängige Stichproben (negativer Bias von ξ und der Risikomaße bei endlichem Stichprobenumfang); zum Vorzeichen des Bias bei serieller Abhängigkeit sagt die Arbeit hier nichts, ein Vergleich mit der negativen Verzerrung dieser Demo wird nicht gezogen.

Keine Neuheit: alle Zahlen sind eigene Nachrechnungen in diesem Modell.

## Verifikation

* **Handbeispiele H 1 bis H 7** als exakte Tests (Brüche, Matrixpotenz, Aufzählung), je mit zweitem Rechenweg.
* **Orakel** (anderer Rechenweg): `tests/test_oracle_theta.py` (θ in Brüchen gegen den vollen Eigenwertlöser, Kern gegen Simulation der Definition und gegen die geschlossene Randverteilung, exakte Blockverteilung des max-AR(1) gegen Simulation, Schätzer an exakten Werten, Pegel); `tests/test_oracle_gpd.py` (GPD-Anpassung gegen `scipy.stats.genpareto` und als Stationärpunkt, Pegel gegen `genpareto.isf`, Delta-Intervall gegen eine Complex-Step-Fassung, die dieselbe Matrix nutzt, **und gegen Simulationen** (Kovarianz von σ̂ und ξ̂, Abdeckung und Streuung an unabhängigen Beobachtungen), eingefrorene GEV gegen `genextreme` mit c = −ξ).
* **Zufall:** `clu_rng.py` ist die wörtliche Kopie der Vorlage (Test `test_rng_stream`); der Strom in Zähler-Form ist bitgleich für die Gleichverteilten, Exponentialzahlen gleich bis 1e-12 relativ.
* **Eingefrorene Daten:** `clu_exact.json`, `clu_blocks.json` (kein Zufall) und `clu_coverage.json` (Läufe); Erzeuger `tools/clu_make_*.py` mit `--write` und `--check`. Die Läufe der Studie und die Live-Läufe der App sind dieselbe Funktion.
* **Eigenschaften, Randfälle, Verdikte, Presets, Oberfläche** (AppTest), **Layout** (Schnellstart, aktives Preset hervorgehoben), **Footer**, **Laufzeit-Importe** (nur streamlit, numpy, pandas, plotly).
* Die Zahlen dieser README werden in `tests/test_claims.py` gegen die Rechnung oder die eingefrorene Datei geprüft (Rundungsband: halbe Einheit der letzten gedruckten Stelle); Laufzeiten, die Zahlen der Vormessung und die Aussagen über Texte (Lesarten, Deutungen) prüft er nicht gegen eine Rechnung. Das Delta-Intervall hat ein unabhängiges Orakel für die Kovarianz nur als Simulation (Band), nicht als Formel.

## Tests

Die Suite hat **393 Tests** (Laufzeit seriell 187 und 252 Sekunden in zwei Läufen, Windows, ein Thread, Maschine unter fremder Last; alle 393 liefen durch, keiner übersprang sich; ohne scipy oder ohne den Nachbarordner `sebastianhanisch-website` im Portfolio überspringen sich einzelne):

* **Handbeispiele** (`test_handbeispiele.py`, 17): H 1 bis H 7 des Plans als exakte Tests (Brüche, Matrixpotenz, Aufzählung).
* **Orakel** (`test_oracle_theta.py` 15, `test_oracle_gpd.py` 54, zusammen 69): siehe Verifikation; darunter Simulationen zum Vorzeichen der Kovarianz und zur Abdeckung (je 3 bis 35 Sekunden).
* **Eigenschaften** (`test_properties.py`, 169) und **Randfälle** (`test_edge_cases.py`, 19).
* **Eingefrorene Daten** (`test_reference.py`, 14) und **Zufall** (`test_rng_stream.py`, 12).
* **Verdikte** (`test_verdicts.py`, 8), **Presets** (`test_presets.py`, 11), **Oberfläche** (`test_app.py`, 21), **Layout** (`test_layout.py`, 28), **Footer** (`test_footer.py`, 2), **Laufzeit-Importe** (`test_runtime_imports.py`, 1).
* **Claims** (`test_claims.py`, 22): die Zahlen dieser README gegen die exakte Rechnung oder die eingefrorene Datei.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Grenzen, Formel-Expander |
| `clu_queue.py` | M/M/1-Wartezeit (Lindley), max-AR(1), Strom in Zähler-Form |
| `clu_kernel.py` | Gitterkern: stationäre Verteilung, Tagesmaximum, θ(u); geschlossene Formeln |
| `clu_theta.py` | Cluster und die drei Schätzer von θ |
| `clu_pot.py` | GPD-Anpassung (ML), Pegel, Delta-Intervall mit θ-Korrektur |
| `clu_evaluation.py` | Auswertungen, Live-Läufe der Abdeckung, Verdikte als reine Funktionen |
| `clu_reference.py` | Zugriff auf die eingefrorenen Messungen |
| `clu_constants.py`, `clu_presets.py`, `clu_layout.py`, `clu_texts.py`, `clu_visualization.py` | Regler, Presets und Permalink, fester Kopf, Zahlenformate, Plotly |
| `clu_rng.py` | wörtliche Kopie des Zufallsgenerators der Vorlage |
| `clu_exact.json`, `clu_blocks.json`, `clu_coverage.json` | eingefrorene Messungen (Quelle, Stand und Software im Kopf) |
| `tools/clu_make_exact.py`, `clu_make_blocks.py`, `clu_make_coverage.py` | Erzeuger der Dateien, je `--write` und `--check` |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly), Entwicklung (pytest, scipy) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, Oberfläche, Footer, Importe |

## Bewusst nicht umgesetzt

Ein Pareto-Bedienzeit-Erzeuger (θ = 0), die Wahl der Schwelle auf Daten und ein PDF-Export. Der Expander „💬 In einfachen Worten“ und die Lesehilfe im 📐-Expander entstehen aus der Erklärseite [Spitzen im Pulk verständlich erklärt](https://sebastianhanisch.net/spitzen-cluster-erklaert.html) (Fachteil: [#fachleute](https://sebastianhanisch.net/spitzen-cluster-erklaert.html#fachleute)).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Extremwerte: Blöcke, Schwelle, Pegel](https://sebastianhanisch.net/konzepte-extremwert.html).
