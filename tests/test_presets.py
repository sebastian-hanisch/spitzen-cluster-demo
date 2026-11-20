"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Verdikt-Zeile je Preset, Permalink (Begrenzen und Einrasten)."""

import pytest

import clu_constants as C
import clu_evaluation as E
import clu_presets as P
import clu_reference as REF

SPEC_OPTIONS = {k: s.options for k, s in P.SETTING_SPECS.items() if s.options}


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_the_first_preset_equals_the_default_settings_and_no_two_presets_are_identical():
    first = next(iter(C.PRESETS.values()))
    assert E.Settings(**first) == E.Settings()
    full = [E.Settings(**p) for p in C.PRESETS.values()]
    assert len({s for s in full}) == len(full)


def test_setting_specs_cover_every_preset_key_and_every_settings_field():
    assert set(P.PRESET_KEYS.values()) | {"seed_input"} == set(P.SETTING_SPECS)
    assert set(E.SETTING_NAMES) - {"seed"} == set(P.PRESET_KEYS)


def test_defaults_lie_on_the_option_lists():
    for key, options in SPEC_OPTIONS.items():
        assert P.SETTING_SPECS[key].default in options


def test_option_regler_have_lo_and_hi_as_extreme_options():
    for key, options in SPEC_OPTIONS.items():
        spec = P.SETTING_SPECS[key]
        assert spec.lo == min(options) and spec.hi == max(options)


@pytest.mark.parametrize("value,options,expected", [(0.79, (0.5, 0.8, 0.9), 0.8), (0.0, (0.5, 0.8, 0.9), 0.5), (7, (16, 64), 16), (1e-3, (1e-4, 1e-3), 1e-3)])
def test_snap_rastet_auf_die_naechste_option_ein(value, options, expected):
    assert P.snap(value, options) == expected


def test_each_preset_shows_the_verdict_of_its_experiment_without_error():
    """Je Preset die erwartete Verdikt-Art und eine Kernzahl (die Seite rechnet in AppTest ohnehin jedes Preset, hier die Verdikte direkt)."""
    s = E.Settings(**C.PRESETS["Last 80 %: 25 je Cluster"])
    a = E.analyse_series(s.rho1, s.n1, s.frac1, s.r1, s.b1, s.seed)
    kind, text = E.verdict_series(a)
    assert "Last 80 %" in text and "θ = 0,0400" in text and "25 Überschreitungen je Cluster" in text
    s = E.Settings(**C.PRESETS["Last 90 %: 100 je Cluster"])
    a = E.analyse_series(s.rho1, s.n1, s.frac1, s.r1, s.b1, s.seed)
    assert "100 Überschreitungen je Cluster" in E.verdict_series(a)[1] and a.n == 20000
    s = E.Settings(**C.PRESETS["Tagesmaximum: unabhängig gerechnet"])
    kind, text = E.verdict_day(E.day_view(s.rho2, s.n2, s.q2))
    assert kind == "warning" and "Last 80 %, 256 Kunden, 0,5-Quantil" in text and "exakt 14,0" in text and "unabhängig gerechnet 28,4" in text
    s = E.Settings(**C.PRESETS["Abdeckung bei Last 80 %"])
    cell = REF.coverage_cell(s.rho3, s.n3, s.frac3, s.p3)
    kind, text = E.verdict_coverage(cell, s.rho3, s.n3, s.frac3, s.p3)
    assert kind == "warning" and "Last 80 %, 3.000 Kunden" in text
    s = E.Settings(**C.PRESETS["Block gegen Schwelle bei Abhängigkeit"])
    assert E.verdict_block_schwelle(REF.block_schwelle_row(s.phi5), s.k5)[0] == "warning"
