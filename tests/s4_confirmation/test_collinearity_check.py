"""
Unit tests for collinearity_check.py
"""
import pytest

from pipeline.config import COLINEAR_ANGLE_MIN
from pipeline.s4_confirmation.collinearity_check import _angle_at_vertex, compute_triplet_angles


def test_angle_at_vertex_straight_line_is_180():
    """
    Checks three points in a straight line give an angle of 180 degrees
    """
    a = (0.0, 0.0)
    v = (5.0, 0.0)
    b = (10.0, 0.0)

    assert _angle_at_vertex(a, v, b) == pytest.approx(180.0)


def test_angle_at_vertex_right_angle_is_90():
    """
    Checks a right-angle triplet gives an angle of 90 degrees
    """
    a = (1.0, 0.0)
    v = (0.0, 0.0)
    b = (0.0, 1.0)

    assert _angle_at_vertex(a, v, b) == pytest.approx(90.0)


def test_angle_at_vertex_degenerate_zero_length_returns_zero():
    """
    Checks a zero-length vector (coincident points) returns 0
    """
    v = (5.0, 5.0)

    assert _angle_at_vertex(v, v, (10.0, 10.0)) == 0.0


def test_compute_triplet_angles_returns_one_row_per_b2_b4_pair():
    """
    Checks one triplet is returned per B2/B4 combination, 
    each correctly flagged as colinear or not
    """
    b2_candidates = [(0.0, 1.0, 10), (0.0, -1.0, 12)]
    b3 = (5.0, 0.0, 20)
    b4_candidates = [(10.0, 1.0, 8), (10.0, -1.0, 9), (10.0, 0.5, 11)]

    triplets = compute_triplet_angles(b2_candidates, b3, b4_candidates)

    assert len(triplets) == len(b2_candidates) * len(b4_candidates)
    for t in triplets:
        assert t["colinear_found"] == (t["angle_deg"] >= COLINEAR_ANGLE_MIN)


def test_compute_triplet_angles_flags_straight_triplet_as_colinear():
    """
    Checks a straight-line triplet is flagged as colinear
    """
    b2_candidates = [(0.0, 0.0, 10)]
    b3 = (5.0, 0.0, 20)
    b4_candidates = [(10.0, 0.0, 8)]

    [triplet] = compute_triplet_angles(b2_candidates, b3, b4_candidates)

    assert triplet["angle_deg"] == pytest.approx(180.0)
    assert triplet["colinear_found"] is True
