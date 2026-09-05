"""
Unit tests for cluster_filter.py
"""
import numpy as np
import pytest

from pipeline.config import CLUSTER_MAX
from pipeline.s2_filter.cluster_filter import _exclude_clusters, _haversine_matrix


def test_haversine_matrix_known_distance():
    """
    Checks the haversine formula gives the correct distance 
    for two points one degree of longitude apart
    """
    R = 6_371_000.0
    lons = np.array([0.0, 1.0])
    lats = np.array([0.0, 0.0])

    dist = _haversine_matrix(lons, lats)

    expected = R * np.radians(1.0)
    assert dist[0, 1] == pytest.approx(expected, rel=1e-6)
    assert dist[1, 0] == pytest.approx(dist[0, 1])


def test_haversine_matrix_zero_distance_to_self():
    """
    Checks a point's distance to itself is zero
    """
    lons = np.array([5.0, 6.0, 7.0])
    lats = np.array([50.0, 51.0, 52.0])

    dist = _haversine_matrix(lons, lats)

    assert np.allclose(np.diag(dist), 0.0)


def test_exclude_clusters_below_threshold_returns_unchanged():
    """
    Checks candidates are returned unchanged when there are too 
    few to check for clusters
    """
    candidates = [{"lon": i * 10.0, "lat": 0.0} for i in range(CLUSTER_MAX)]

    assert _exclude_clusters(candidates) == candidates


def test_exclude_clusters_removes_dense_cluster_keeps_isolated_point():
    """
    Checks a dense cluster is excluded while an isolated candidate survives
    """
    dense_cluster = [
        {"lon": 4.0 + i * 0.0001, "lat": 52.0, "id": f"dense-{i}"}
        for i in range(CLUSTER_MAX + 1)
    ]
    isolated = {"lon": 10.0, "lat": 52.0, "id": "isolated"}

    survivors = _exclude_clusters(dense_cluster + [isolated])

    assert survivors == [isolated]
