"""
Unit tests for displacement_check.py
"""
import math

import pytest

from pipeline.config import (
    DISPLACEMENT_OFFSET,
    DISPLACEMENT_RATIO,
    DISPLACEMENT_TOLERANCE,
    METRES_PER_PIXEL,
    T_BAND2_4,
)
from pipeline.s4_confirmation.displacement_check import _pixel_dist, check_displacement


def test_pixel_dist_zero_for_same_point():
    """Checks the distance between a point and itself is zero."""
    assert _pixel_dist(10, 20, 10, 20) == 0.0


def test_pixel_dist_matches_3_4_5_triangle():
    """Checks the pixel distance calculation matches a simple 3-4-5 triangle."""
    dist = _pixel_dist(0, 0, 3, 4)

    assert dist == pytest.approx(5 * METRES_PER_PIXEL)


def _triplet_for_d23(d23_m, d34_m, angle_deg=175.0):
    """
    Builds a collinearity_triplet-shaped dict with exact D_23/D_34
    pixel distances (in metres), laid out along a straight pixel row,
    for check_displacement().
    """
    col23 = d23_m / METRES_PER_PIXEL
    col34 = d34_m / METRES_PER_PIXEL
    return {
        'c_b2_col': 0.0, 'c_b2_row': 0.0,
        'c_b3_col': col23, 'c_b3_row': 0.0,
        'c_b4_col': col23 + col34, 'c_b4_row': 0.0,
        'angle_deg': angle_deg,
    }


def test_check_displacement_empty_triplets_returns_rejected():
    """Checks an empty triplet list is rejected with no valid result."""
    result = check_displacement([])

    assert result['confirmed'] is False
    assert math.isnan(result['residual_m'])


def test_check_displacement_confirms_exact_match():
    """Checks a triplet with an exact displacement match is confirmed."""
    d23 = 200.0
    d34_expected = DISPLACEMENT_RATIO * d23 + DISPLACEMENT_OFFSET
    d24 = d23 + d34_expected
    triplet = _triplet_for_d23(d23, d34_expected)

    result = check_displacement([triplet])

    assert result['confirmed'] is True
    assert result['residual_m'] == pytest.approx(0.0, abs=1e-4)
    assert result['D_Band2_3_m'] == pytest.approx(d23)
    assert result['speed_kmh'] == pytest.approx((d24 / T_BAND2_4) * 3.6)


def test_check_displacement_rejects_large_residual():
    """
    Checks a triplet with too large a residual is rejected
    """
    d23 = 200.0
    d34_expected = DISPLACEMENT_RATIO * d23 + DISPLACEMENT_OFFSET
    d34_way_off = d34_expected + DISPLACEMENT_TOLERANCE * 10
    triplet = _triplet_for_d23(d23, d34_way_off)

    result = check_displacement([triplet])

    assert result['confirmed'] is False
    assert result['residual_m'] > DISPLACEMENT_TOLERANCE
    assert not math.isnan(result['residual_m'])


def test_check_displacement_picks_lowest_residual_among_passing_triplets():
    """
    Checks the triplet with the smallest residual is picked when several pass
    """
    d23 = 200.0
    d34_expected = DISPLACEMENT_RATIO * d23 + DISPLACEMENT_OFFSET
    close  = _triplet_for_d23(d23, d34_expected + 1.0)    # residual ~1.0
    closer = _triplet_for_d23(d23, d34_expected + 0.1)    # residual ~0.1

    result = check_displacement([close, closer])

    assert result['confirmed'] is True
    assert result['residual_m'] == pytest.approx(0.1, abs=1e-3)
