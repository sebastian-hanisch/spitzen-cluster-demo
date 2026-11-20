"""Zahlenformate der App: Dezimalkomma, Tausenderpunkt, echtes Minuszeichen. Zahlen in Texten werden aus der Rechnung oder der eingefrorenen Referenz gebildet, nie von Hand eingetragen."""

import math

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def de(x: float, nd: int = 2) -> str:
    """Dezimalzahl mit Komma: de(0.0213, 4) == '0,0213'; negative Zahlen mit echtem Minuszeichen."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def de_signed(x: float, nd: int = 2) -> str:
    """Mit Vorzeichen und echtem Minuszeichen: de_signed(-0.02, 3) == '−0,020'."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def thousands(x: float) -> str:
    """Ganzzahl mit Tausenderpunkt: 20000 -> '20.000'."""
    return f"{int(round(x)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozent mit Komma: pct(0.325) == '32,5 %'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{nd}f}".replace(".", ",") + " %"


def rho_text(rho: float) -> str:
    """Last als Prozent: 0.8 -> '80 %'."""
    return f"{100 * rho:g}".replace(".", ",") + " %"


def p_text(p: float) -> str:
    """Überschreitungswahrscheinlichkeit: 0,001 → '10⁻³', sonst mit Komma."""
    e = math.log10(p)
    if abs(e - round(e)) < 1e-9:
        return "10" + str(int(round(e))).translate(_SUP)
    return f"{p:.6f}".rstrip("0").rstrip(".").replace(".", ",")


def frac_text(f: float) -> str:
    """Anteil der Kunden über der Schwelle in Prozent ohne überflüssige Stellen: 0.05 → '5 %', 0.001 → '0,1 %'."""
    return f"{100 * f:g}".replace(".", ",") + " %"


def minutes(x_service_times: float, per: float = 3.0) -> str:
    """Abfertigungsdauern als Minuten bei einer angenommenen Abfertigungsdauer von `per` Minuten (reine Umrechnung)."""
    return f"{x_service_times * per:.0f}".replace(".", ",") + " min"


def fraction_text(num: int, den: int) -> str:
    """Bruch als Text: '1/25'."""
    return f"{num}/{den}"


def q_text(q: float) -> str:
    """Quantil ohne überflüssige Stellen: 0.5 → '0,5', 0.99 → '0,99', 0.999 → '0,999'."""
    return f"{q:g}".replace(".", ",")
