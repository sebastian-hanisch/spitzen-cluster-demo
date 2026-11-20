"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import clu_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # bei Auswahl-Reglern: erlaubte Werte (der Permalink rastet auf den nächsten ein)


SETTING_SPECS = {
    "rho1": SettingSpec("rho1", float, C.DEFAULT_RHO1, C.RHO1_MIN, C.RHO1_MAX),
    "n1": SettingSpec("n1", int, C.DEFAULT_N1, C.N1_MIN, C.N1_MAX),
    "frac1": SettingSpec("frac1", float, C.DEFAULT_FRAC1, min(C.FRAC1_OPTIONS), max(C.FRAC1_OPTIONS), C.FRAC1_OPTIONS),
    "r1": SettingSpec("r1", int, C.DEFAULT_R1, C.R1_MIN, C.R1_MAX),
    "b1": SettingSpec("b1", int, C.DEFAULT_B1, C.B1_MIN, C.B1_MAX),
    "rho2": SettingSpec("rho2", float, C.DEFAULT_RHO2, min(C.RHO2_OPTIONS), max(C.RHO2_OPTIONS), C.RHO2_OPTIONS),
    "n2": SettingSpec("n2", int, C.DEFAULT_N2, min(C.N2_OPTIONS), max(C.N2_OPTIONS), C.N2_OPTIONS),
    "q2": SettingSpec("q2", float, C.DEFAULT_Q2, min(C.Q2_OPTIONS), max(C.Q2_OPTIONS), C.Q2_OPTIONS),
    "rho3": SettingSpec("rho3", float, C.DEFAULT_RHO3, min(C.RHO3_OPTIONS), max(C.RHO3_OPTIONS), C.RHO3_OPTIONS),
    "n3": SettingSpec("n3", int, C.DEFAULT_N3, min(C.N3_OPTIONS), max(C.N3_OPTIONS), C.N3_OPTIONS),
    "frac3": SettingSpec("frac3", float, C.DEFAULT_FRAC3, min(C.FRAC3_OPTIONS), max(C.FRAC3_OPTIONS), C.FRAC3_OPTIONS),
    "p3": SettingSpec("p3", float, C.DEFAULT_P3, min(C.P3_OPTIONS), max(C.P3_OPTIONS), C.P3_OPTIONS),
    "r3": SettingSpec("r3", int, C.DEFAULT_R3, min(C.R3_OPTIONS), max(C.R3_OPTIONS), C.R3_OPTIONS),
    "rho4": SettingSpec("rho4", float, C.DEFAULT_RHO4, min(C.RHO4_OPTIONS), max(C.RHO4_OPTIONS), C.RHO4_OPTIONS),
    "n4": SettingSpec("n4", int, C.DEFAULT_N4, min(C.N4_OPTIONS), max(C.N4_OPTIONS), C.N4_OPTIONS),
    "phi5": SettingSpec("phi5", float, C.DEFAULT_PHI5, min(C.PHI5_OPTIONS), max(C.PHI5_OPTIONS), C.PHI5_OPTIONS),
    "k5": SettingSpec("k5", int, C.DEFAULT_K5, min(C.K5_OPTIONS), max(C.K5_OPTIONS), C.K5_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS if key != "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap(value, options):
    """Der erlaubte Wert, der `value` am nächsten liegt (Permalinks mit fremden Werten rasten ein, statt die App zu stören)."""
    return min(options, key=lambda o: abs(o - value))


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf die Auswahl eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(float(qp[spec.url_param]) if spec.caster is int else qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None:
                value = snap(value, spec.options)
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: genannte auf ihren Wert, alle übrigen (auch der Zufalls-Seed) auf den Standard, sonst bliebe das aktive Preset unklar."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Der Zufalls-Seed gehört dazu (er ändert die Reihe aus Experiment 1); ein Preset stellt ihn auf den Standard."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(PRESET_KEYS.get(state_key), spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
