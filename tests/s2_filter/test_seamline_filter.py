"""
Unit tests for seamline_filter.py
"""
import numpy as np
import pytest

from pipeline.config import SEAM_MIN_INLIERS
from pipeline.s2_filter.seamline_filter import (
    _fit_line_normal_form,
    _line_angle_deg,
    _msac_best_line,
    _point_line_distance,
)


def test_fit_line_normal_form_duplicate_points_returns_none():
    """
    Checks fitting a line through two identical 
    points returns None
    """
    theta, rho = _fit_line_normal_form((1.0, 2.0), (1.0, 2.0))

    assert theta is None
    assert rho is None


def test_fit_line_normal_form_satisfies_line_equation_for_both_points():
    """
    Checks the fitted line's equation holds true for both input points.
    """
    p1 = (0.0, 0.0)
    p2 = (100.0, 50.0)

    theta, rho = _fit_line_normal_form(p1, p2)

    assert theta is not None
    for x, y in (p1, p2):
        assert x * np.cos(theta) + y * np.sin(theta) == pytest.approx(rho, abs=1e-6)


def test_point_line_distance_zero_for_points_on_line():
    """
    Checks points that lie exactly on the line have zero distance to it
    """
    theta, rho = _fit_line_normal_form((0.0, 0.0), (100.0, 0.0))
    pts = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])

    dist = _point_line_distance(pts, theta, rho)

    assert np.allclose(dist, 0.0, atol=1e-6)


def test_point_line_distance_matches_known_perpendicular_offset():
    """
    Checks the distance calculation matches a known perpendicular offset
    """
    theta, rho = _fit_line_normal_form((0.0, 0.0), (100.0, 0.0))
    pts = np.array([[50.0, 10.0]])

    dist = _point_line_distance(pts, theta, rho)

    assert dist[0] == pytest.approx(10.0)


def test_line_angle_deg_is_perpendicular_to_normal():
    """
    Checks the line's angle is 90 degrees away from its normal vector's angle.
    """
    assert _line_angle_deg(0.0) == pytest.approx(90.0)


def test_msac_best_line_finds_aligned_points_as_inliers():
    """
    Checks MSAC picks all aligned points as inliers and excludes a clear outlier.
    """
    n_inliers = SEAM_MIN_INLIERS + 3
    line_pts = np.array([[float(i) * 100, 0.0] for i in range(n_inliers)])
    outlier = np.array([[500.0, 5000.0]])
    pts = np.vstack([line_pts, outlier])
    outlier_idx = pts.shape[0] - 1

    result = _msac_best_line(pts)

    assert result is not None
    _theta, _rho, inliers = result
    assert len(inliers) == n_inliers
    assert outlier_idx not in inliers


def test_msac_best_line_returns_none_below_min_inliers():
    """
    Checks MSAC returns None when there are too few points to 
    reach the minimum inlier count
    """
    assert SEAM_MIN_INLIERS > 2
    pts = np.array([[0.0, 0.0], [100.0, 0.0]])

    assert _msac_best_line(pts) is None
