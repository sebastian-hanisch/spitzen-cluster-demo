"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen und prüft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig identischen Inhalt rendern und
kollidieren) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max).

Der erste Teil ist die Vorlage des Portfolios (tools/templates/test_app_smoke.py); sie zählt nur `st.slider`. Diese Demo hat überwiegend `st.select_slider`: der Zusatztest unten prüft jeden
Auswahl-Regler an seiner ersten und letzten Option, jede Option der Verdikt-Regler und den Inhalt der Seite je Preset."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import clu_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusatz: Auswahl-Regler und Inhalt -------------------------------------------------------------------------------------------------------

def test_every_select_slider_at_its_first_and_last_option_does_not_raise():
    n = len(_fresh_app().select_slider)
    assert n == 13
    for index in range(n):
        for edge in (0, -1):
            at = _fresh_app()
            sl = at.select_slider[index]
            sl.set_value(OPTIONS[sl.key][edge]).run(timeout=TIMEOUT)
            assert not at.exception, (sl.label, edge, [str(e) for e in at.exception])


OPTIONS = {"rho2": C.RHO2_OPTIONS, "n2": C.N2_OPTIONS, "q2": C.Q2_OPTIONS, "rho3": C.RHO3_OPTIONS, "n3": C.N3_OPTIONS, "frac3": C.FRAC3_OPTIONS, "p3": C.P3_OPTIONS,
           "r3": C.R3_OPTIONS, "rho4": C.RHO4_OPTIONS, "n4": C.N4_OPTIONS, "phi5": C.PHI5_OPTIONS, "k5": C.K5_OPTIONS, "frac1": C.FRAC1_OPTIONS}


@pytest.mark.parametrize("key", list(OPTIONS))
def test_every_option_of_every_select_slider_renders(key):
    at = _fresh_app()
    for opt in OPTIONS[key]:
        at.sidebar.select_slider(key=key).set_value(opt).run(timeout=TIMEOUT)
        assert not at.exception, (key, opt, [str(e) for e in at.exception])
        assert at.sidebar.select_slider(key=key).value == opt


def test_jeder_preset_zeigt_seine_verdikte_und_keine_exception():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        boxes = [x.value for kind in ("success", "info", "warning") for x in getattr(at, kind)]
        assert len(boxes) == 6, (name, len(boxes))                                    # je Experiment ein Verdikt, dazu das der Schätzer-Tabelle
        assert at.sidebar.slider(key="rho1").value == C.PRESETS[name].get("rho1", C.DEFAULT_RHO1)


def test_die_seite_hat_fuenf_experimente_die_grenzen_und_den_formel_expander():
    at = _fresh_app()
    heads = [s.value for s in at.subheader]
    assert [h for h in heads if h.startswith("🔬 Experiment")] == [f"🔬 Experiment {i}: {t}" for i, t in enumerate(
        ["Eine Reihe, ihre Spitzen, ihre Cluster", "Das Tagesmaximum, unabhängig gerechnet gegen exakt", "Hält das 95-%-Intervall sein Versprechen?", "Wie groß muss der Block sein?",
         "Block gegen Schwelle, wenn die Werte voneinander abhängen"], start=1)]
    assert "🚧 Wo die Annahmen enden" in heads
    labels = [e.label for e in at.expander]
    assert labels[0] == "💬 In einfachen Worten" and "So funktioniert die Seite" in labels
    assert "📐 Mathematische Formulierung" in labels and labels.index("📐 Mathematische Formulierung") == len(labels) - 1
    assert len(at.get("plotly_chart")) >= 6


def test_live_laeufe_aus_und_an_aendern_die_seite_nur_um_die_live_zeile():
    at = _fresh_app()
    n_on = len([c for c in at.caption if c.value.startswith("Live:")])
    at.sidebar.select_slider(key="r3").set_value(0).run(timeout=TIMEOUT)
    assert not at.exception and not [c for c in at.caption if c.value.startswith("Live:")] and n_on == 1


def test_permalink_rastet_ein_und_begrenzt():
    at = AppTest.from_file(APP_PATH)
    at.query_params["rho1"] = "7.5"          # über der Grenze: begrenzt auf 0,9
    at.query_params["rho2"] = "0.79"         # nicht auf der Liste: rastet auf 0,8 ein
    at.query_params["n1"] = "5"              # unter der Grenze: begrenzt auf 300
    at.query_params["p3"] = "abc"            # unlesbar: Standard bleibt
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.slider(key="rho1").value == C.RHO1_MAX and at.sidebar.select_slider(key="rho2").value == 0.8
    assert at.sidebar.slider(key="n1").value == C.N1_MIN and at.sidebar.select_slider(key="p3").value == C.DEFAULT_P3
