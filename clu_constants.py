"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Vorzeichen der Linie: ξ > 0 schwerer Rand (Fréchet-Typ), ξ = 0 Gumbel-Typ, ξ < 0 endlicher Endpunkt; `scipy.stats.genextreme` hat c = −ξ, `genpareto` c = ξ (nur in Tests).
θ ist der Extremalindex, ξ der Extremwertindex: nie vermischen."""

SEED_MAX, DEFAULT_SEED = 999999, 2026

# Experiment 1: eine Reihe, ihre Spitzen, ihre Cluster
RHO1_MIN, RHO1_MAX, RHO1_STEP, DEFAULT_RHO1 = 0.30, 0.90, 0.05, 0.50
N1_MIN, N1_MAX, N1_STEP, DEFAULT_N1 = 300, 20000, 100, 3000
FRAC1_OPTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
DEFAULT_FRAC1 = 0.05
R1_MIN, R1_MAX, DEFAULT_R1 = 1, 50, 10
B1_MIN, B1_MAX, B1_STEP, DEFAULT_B1 = 10, 200, 10, 50

# Experiment 2: das Tagesmaximum (eingefroren, exakt)
RHO2_OPTIONS = (0.5, 0.8, 0.9)
N2_OPTIONS = (16, 32, 64, 128, 256, 512, 1024)
Q2_OPTIONS = (0.1, 0.5, 0.9, 0.99, 0.999)
DEFAULT_RHO2, DEFAULT_N2, DEFAULT_Q2 = 0.5, 64, 0.5

# Experiment 3: hält das 95-%-Intervall? (eingefrorene Zellen, dazu Läufe live)
RHO3_OPTIONS = (0.5, 0.8)
N3_OPTIONS = (1000, 3000, 12000)
FRAC3_OPTIONS = (0.02, 0.05, 0.10)
P3_OPTIONS = (1e-4, 1e-3)
R3_OPTIONS = (0, 20, 50, 100, 200)       # 0 = keine Live-Läufe
DEFAULT_RHO3, DEFAULT_N3, DEFAULT_FRAC3, DEFAULT_P3, DEFAULT_R3 = 0.5, 1000, 0.10, 1e-3, 20
LIVE_N_MAX_PRODUCT = 200 * 12000           # Obergrenze Läufe × Kunden der Live-Läufe (Rechenzeit, siehe README)

# Experiment 4: wie groß muss der Block sein? (eingefroren, exakt)
RHO4_OPTIONS = (0.5, 0.8)
N4_OPTIONS = (64, 256, 1024, 4096)
DEFAULT_RHO4, DEFAULT_N4 = 0.8, 256

# Experiment 5: Block gegen Schwelle unter Abhängigkeit am max-AR(1) (eingefroren)
PHI5_OPTIONS = (0.0, 0.5, 0.8)
K5_OPTIONS = (50, 125)
DEFAULT_PHI5, DEFAULT_K5 = 0.0, 125

DAY_MIN_MINUTES = 3.0                      # nur Umrechnung: 1 Abfertigungsdauer = 3 min (Annahme, kein Befund)
MIN_CLUSTERS_OK = 10                       # darunter nennt das Verdikt „zu wenige Cluster“ (Schwelle der Seite, keine Eigenschaft der Methode)

COLOR_EXACT = "#2F6B65"
COLOR_THETA = "#d95f02"
COLOR_IID = "#7570b3"
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.55)"
CLUSTER_COLORS = ("#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666")

PRESETS = {
    "Last 50 %: 4 je Cluster": {},
    "Last 80 %: 25 je Cluster": {"rho1": 0.80},
    "Last 90 %: 100 je Cluster": {"rho1": 0.90, "n1": 20000, "frac1": 0.01},
    "Tagesmaximum: unabhängig gerechnet": {"rho2": 0.8, "n2": 256, "q2": 0.5},
    "Abdeckung bei Last 80 %": {"rho3": 0.8, "n3": 3000, "frac3": 0.05, "p3": 1e-3, "r3": 100},
    "Block gegen Schwelle bei Abhängigkeit": {"phi5": 0.8, "k5": 50},
}
PRESET_HELP = {
    "Last 50 %: 4 je Cluster": "M/M/1-Wartezeit bei Last 50 %, 3.000 Kunden, Schwelle bei 5 % der Kunden: Der Extremalindex ist 1/4, ein Cluster hat im Grenzwert im Mittel 4 Überschreitungen.",
    "Last 80 %: 25 je Cluster": "Last 80 %: θ = 1/25, ein Cluster hat im Grenzwert im Mittel 25 Überschreitungen. Bei 150 Überschreitungen stecken rechnerisch nur 6 unabhängige Cluster darin.",
    "Last 90 %: 100 je Cluster": "Last 90 %, 20.000 Kunden, Schwelle bei 1 % der Kunden: θ = 1/100, im Grenzwert im Mittel 100 Überschreitungen je Cluster.",
    "Tagesmaximum: unabhängig gerechnet": "Last 80 %, ein Tag mit 256 Kunden: das exakte Tagesmaximum gegen die Rechnung „als wären die Kunden unabhängig“.",
    "Abdeckung bei Last 80 %": "Last 80 %, 3.000 Kunden, 95-%-Intervall für den Pegel zu p = 10⁻³ je Kunde: Wie oft liegt der wahre Pegel im Intervall, wenn die Abhängigkeit übergangen wird?",
    "Block gegen Schwelle bei Abhängigkeit": "max-AR(1) mit φ = 0,8 (θ = 0,2): Block-GEV gegen Schwellenweg bei 50 Überschreitungen; die eingefrorenen Läufe zeigen, was die θ-Korrektur leistet.",
}
