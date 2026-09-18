"""Tests for active-learning-ready image selection."""

from __future__ import annotations

import pytest

from NEBwalk.selection import select_images, select_peak_plus_neighbors


def test_selects_peak_plus_neighbors():
    energies = [0.0, 0.1, 0.3, 0.6, 0.4, 0.2, 0.0]

    assert select_peak_plus_neighbors(energies, n_select=3) == [2, 3, 4]


def test_excludes_endpoints_by_default():
    energies = [10.0, 1.0, 2.0, 1.5, 9.0]
    selected = select_peak_plus_neighbors(energies, n_select=2)

    assert 0 not in selected
    assert 4 not in selected


def test_can_include_endpoints():
    energies = [10.0, 1.0, 2.0]

    assert 0 in select_peak_plus_neighbors(
        energies,
        n_select=1,
        include_endpoints=True,
    )


def test_fills_extra_selections_by_next_highest_energy():
    energies = [0.0, 0.5, 0.2, 1.0, 0.3, 0.9, 0.0]

    assert select_peak_plus_neighbors(energies, n_select=5) == [1, 2, 3, 4, 5]


def test_returns_all_when_fewer_eligible_than_requested():
    energies = [0.0, 1.0, 0.0]

    assert select_peak_plus_neighbors(energies, n_select=3) == [1]


def test_raises_for_empty_energies():
    with pytest.raises(ValueError, match="at least one"):
        select_peak_plus_neighbors([])


def test_raises_for_invalid_n_select():
    with pytest.raises(ValueError, match="n_select"):
        select_peak_plus_neighbors([0.0, 1.0, 0.0], n_select=0)


def test_raises_when_no_eligible_images():
    with pytest.raises(ValueError, match="no eligible"):
        select_peak_plus_neighbors([0.0, 1.0])


def test_raises_for_nonfinite_energies():
    with pytest.raises(ValueError, match="finite"):
        select_peak_plus_neighbors([0.0, float("nan"), 0.0])


def test_dispatcher_rejects_unsupported_strategy():
    with pytest.raises(ValueError, match="unsupported selection strategy"):
        select_images([0.0, 1.0, 0.0], strategy="uncertainty_guided")
