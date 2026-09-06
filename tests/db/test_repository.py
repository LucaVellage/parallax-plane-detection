"""
Unit tests for repository.py
"""
import pytest

from pipeline.db.repository import _find_closest_chip


def test_find_closest_chip_empty_rows_returns_none():
    """
    Checks an empty cache (no rows for this tile/date) is a clean miss
    """
    assert _find_closest_chip([], col=100.0, row=200.0, tol_px=0.6) is None


def test_find_closest_chip_exact_match():
    """
    Checks a cached row at the exact requested position is returned
    """
    rows = [{"col": 100.0, "row": 200.0, "file_path": "a.tif", "checksum": "abc"}]

    match = _find_closest_chip(rows, col=100.0, row=200.0, tol_px=0.6)

    assert match == rows[0]


def test_find_closest_chip_within_tolerance():
    """
    Checks a cached row just inside the pixel tolerance still counts as a match
    """
    rows = [{"col": 100.0, "row": 200.0, "file_path": "a.tif", "checksum": "abc"}]

    match = _find_closest_chip(rows, col=100.3, row=200.2, tol_px=0.6)

    assert match == rows[0]


def test_find_closest_chip_outside_tolerance_returns_none():
    """
    Checks a cached row further than the tolerance is treated as a miss,
    not silently reused
    """
    rows = [{"col": 100.0, "row": 200.0, "file_path": "a.tif", "checksum": "abc"}]

    match = _find_closest_chip(rows, col=105.0, row=200.0, tol_px=0.6)

    assert match is None


def test_find_closest_chip_picks_nearest_of_several():
    """
    Checks the closest candidate row wins when several are within tolerance
    """
    near = {"col": 100.1, "row": 200.1, "file_path": "near.tif", "checksum": "n"}
    far  = {"col": 100.5, "row": 200.5, "file_path": "far.tif", "checksum": "f"}

    match = _find_closest_chip([far, near], col=100.0, row=200.0, tol_px=0.6)

    assert match == near
