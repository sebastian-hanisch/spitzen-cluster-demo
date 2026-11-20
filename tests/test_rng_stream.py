"""Die kopierte Datei clu_rng.py (wörtlich, Quelle im Kopf der Datei) und der Strom in Zähler-Form (`clu_queue`): bitgleiche Gleichverteilte, Exponentialzahlen bis 1e-12 relativ."""

from pathlib import Path

import numpy as np
import pytest

import clu_queue as Q
from clu_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def test_kopie_ist_woertlich_bis_auf_die_kopfzeile():
    if not SOURCE.exists():
        pytest.skip("Vorlage liegt nicht im Portfolio-Ordner")
    mine = (ROOT / "clu_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith("# Wörtliche Kopie von sebastianhanisch-website/tools/templates/rng_splitmix64_exact.py")
    assert mine[1:] == SOURCE.read_text(encoding="utf-8").splitlines(), "die Kopie weicht von der Vorlage ab: Änderungen gehören in die Vorlage, danach neu kopieren"


def test_feste_anker_der_vorlage():
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    r = SplitMix64(42)
    assert [r.random().hex() for _ in range(2)] == ["0x1.7bae644c5fd6dp-1", "0x1.477f199d93378p-3"]


def test_u64_5000_ausgaenge_wert_fuer_wert():
    r = SplitMix64(13)
    assert [r.next_u64() for _ in range(5000)] == Q.sm64_u64(13, 5000).tolist()


def test_random_3000_werte_exakt():
    r = SplitMix64(7)
    assert [r.random() for _ in range(3000)] == Q.sm64_unif(7, 3000).tolist()


def test_strom_aus_zwei_stuecken_gleich_dem_strom_am_stueck():
    assert np.array_equal(np.concatenate([Q.sm64_u64(5, 100), Q.sm64_u64(5, 50, 100)]), Q.sm64_u64(5, 150))


def test_exponentialzahlen_gleich_expovariate_der_vorlage_bis_1e_12():
    r = SplitMix64(99)
    pure = np.array([r.expovariate(2.5) for _ in range(500)])
    vec = Q.exp_from_u(Q.sm64_unif(99, 500), 2.5)
    assert np.allclose(pure, vec, rtol=1e-12, atol=0)


@pytest.mark.parametrize("rho,n,run", [(0.5, 400, 0), (0.8, 400, 3), (0.9, 150, 7)])
def test_vektor_reihe_gleich_der_schleife_nach_der_definition(rho, n, run):
    a = Q.mm1_wait(rho, n, 17, run)
    b = np.array(Q.mm1_wait_loop(rho, n, 17, run))
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_laeufe_desselben_seeds_sind_verschiedene_stuecke_des_stroms_und_wiederholbar():
    a, b = Q.mm1_wait(0.8, 200, 5, 0), Q.mm1_wait(0.8, 200, 5, 1)
    assert not np.allclose(a, b)
    assert np.array_equal(Q.mm1_wait(0.8, 200, 5, 1), b)
    assert not np.allclose(Q.mm1_wait(0.8, 200, 5, 0), Q.mm1_wait(0.8, 200, 6, 0))


def test_die_drei_stroeme_eines_seeds_ueberlappen_nicht():
    s = Q.stream_seeds(2026)
    assert len(set(s)) == 3
    assert not np.any(np.isin(Q.sm64_u64(s[0], 5000), Q.sm64_u64(s[1], 5000)))


def test_reihe_startet_stationaer_ohne_einschwingzeit():
    first = np.array([Q.mm1_wait(0.8, 1, 11, r)[0] for r in range(6000)])
    assert abs(float(np.mean(first > 0)) - 0.8) < 4 * np.sqrt(0.8 * 0.2 / 6000)
    assert abs(float(first.mean()) - 4.0) < 0.35                                   # E[W] = ρ/(1−ρ) = 4 bei Last 0,8 (nicht 0 wie bei leerem Start)
