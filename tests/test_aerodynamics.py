"""
Basic unit tests for the aerodynamics module.
Run with:  pytest tests/test_aerodynamics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aerodynamics import (
    BoattailGeometry,
    LauncherGeometry,
    _wave_drag_transition_bounds,
    angle_of_attack_increment,
    compute_drag,
    _atmosphere,
)


# ---- helpers ---------------------------------------------------------------

def simple_geom(**kwargs):
    """Return a minimal valid LauncherGeometry."""
    defaults = dict(
        stage_lengths=[1.8, 7.0, 3.5],
        stage_diameters=[1.2, 1.3, 1.0],
    )
    defaults.update(kwargs)
    return LauncherGeometry(**defaults)


# ---- atmosphere ------------------------------------------------------------

def test_atmosphere_sea_level():
    a, nu = _atmosphere(0.0)
    # ISA sea-level: ~340 m/s, ~1.46e-5 m²/s
    assert 330 < a < 350
    assert 1e-5 < nu < 2e-5


def test_atmosphere_stratosphere():
    a, nu = _atmosphere(15_000)
    assert 295 < a < 320   # speed of sound drops in the stratosphere


@pytest.mark.parametrize(
    ("altitude_m", "expected_a", "expected_nu"),
    [
        (0.0, 340.3, 1.46e-5),
        (11_000.0, 295.2, 3.9e-5),
        (20_000.0, 295.1, 1.61e-4),
    ],
)
def test_atmosphere_matches_isa_reference_points(altitude_m, expected_a, expected_nu):
    a, nu = _atmosphere(altitude_m)
    assert a == pytest.approx(expected_a, rel=0.02)
    assert nu == pytest.approx(expected_nu, rel=0.08)


# ---- derived geometry ------------------------------------------------------

def test_geometry_totals():
    g = simple_geom()
    assert abs(g.L_total - (1.8 + 7.0 + 3.5)) < 1e-9
    assert g.d_max == 1.3
    assert g.S_wet_total > 0


def test_geometry_requires_matching_stage_vectors():
    with pytest.raises(ValueError, match="same length"):
        LauncherGeometry(
            stage_lengths=[1.8, 7.0, 3.5],
            stage_diameters=[1.2, 1.3],
        )


def test_geometry_requires_boattail_definition_when_enabled():
    with pytest.raises(ValueError, match="Boattail geometry must be provided"):
        simple_geom(boattail_exists=True, boattail=None)


# ---- component drag sanity -------------------------------------------------

def test_zero_lift_drag_positive():
    mach = np.arange(0.1, 6.1, 0.1)
    g = simple_geom()
    result = compute_drag(g, mach)
    assert np.all(result.CD_total_zero_lift >= 0), "CD must be non-negative"
    assert np.all(result.Cd_friction_total  >= 0)
    assert np.all(result.Cd_base           >= 0)


def test_zero_lift_drag_shape():
    mach = np.arange(0.1, 4.1, 0.1)
    g = simple_geom()
    result = compute_drag(g, mach)
    assert result.CD_total_zero_lift.shape == mach.shape


def test_boattail_increases_drag():
    """Adding a boattail should not decrease total drag across most of the range."""
    mach = np.arange(0.9, 4.0, 0.1)  # transonic/supersonic
    g_no_bt = simple_geom(boattail_exists=False)
    bt = BoattailGeometry(length=0.6, diameter_fore=1.0,
                          diameter_aft=1.3, ref_diameter=1.3)
    g_bt = simple_geom(boattail_exists=True, boattail=bt)

    r_no = compute_drag(g_no_bt, mach)
    r_bt = compute_drag(g_bt, mach)
    # boattail adds drag for M > 0.8
    assert np.sum(r_bt.CD_total_zero_lift) >= np.sum(r_no.CD_total_zero_lift)


def test_wave_drag_subsonic_small():
    """Wave drag should be negligible well below divergence Mach."""
    mach = np.array([0.1, 0.2, 0.3])
    g = simple_geom()
    result = compute_drag(g, mach)
    assert np.all(result.CD_wave_supersonic[:3] == 0)


def test_wave_drag_supersonic_nonzero():
    """Wave drag must be present at clearly supersonic speeds."""
    mach = np.array([2.0, 3.0, 4.0, 5.0])
    g = simple_geom()
    result = compute_drag(g, mach)
    assert np.all(result.CD_wave_supersonic > 0)


def test_base_drag_is_independent_of_mach_grid_anchor_sampling():
    g = simple_geom()
    mach_dense = np.array([0.6, 0.9, 1.2, 2.0, 3.0])
    mach_sparse = np.array([0.9, 1.2, 2.0, 3.0])

    result_dense = compute_drag(g, mach_dense)
    result_sparse = compute_drag(g, mach_sparse)

    np.testing.assert_allclose(
        result_dense.Cd_base[1:],
        result_sparse.Cd_base,
        rtol=1e-10,
        atol=1e-10,
    )


def test_wave_drag_transition_is_continuous():
    g = simple_geom()
    mach_div, mach_final = _wave_drag_transition_bounds(g)
    eps = 1e-4
    mach = np.array([
        mach_div - eps,
        mach_div,
        mach_div + eps,
        mach_final - eps,
        mach_final,
        mach_final + eps,
    ])

    result = compute_drag(g, mach)
    wave_total = result.CD_wave_transonic + result.CD_wave_supersonic

    assert abs(wave_total[1] - wave_total[0]) < 1e-3
    assert abs(wave_total[2] - wave_total[1]) < 1e-3
    assert abs(wave_total[4] - wave_total[3]) < 1e-3
    assert abs(wave_total[5] - wave_total[4]) < 1e-3


# ---- angle-of-attack increment ---------------------------------------------

def test_aoa_increment_zero_at_zero():
    g = simple_geom()
    dCD = angle_of_attack_increment(g, np.array([0.0]))
    assert abs(dCD[0]) < 1e-10


def test_aoa_increment_positive():
    g = simple_geom()
    alphas = np.array([5.0, 10.0, 15.0])
    dCD = angle_of_attack_increment(g, alphas)
    assert np.all(dCD >= 0)
    # larger AoA → larger increment
    assert dCD[1] > dCD[0]
    assert dCD[2] > dCD[1]


# ---- custom Mach array -----------------------------------------------------

def test_single_mach_point():
    g = simple_geom()
    result = compute_drag(g, np.array([1.5]))
    assert result.CD_total_zero_lift.shape == (1,)
    assert result.CD_total_zero_lift[0] > 0
