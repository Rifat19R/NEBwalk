"""Committee disagreement and diverse candidate-selection tests."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from nebwalk.candidate_selection import (
    CandidateInput,
    CandidateSelectionConfig,
    select_active_learning_candidates,
)
from nebwalk.mlip import (
    CommitteeCalculator,
    CommitteeEvaluationError,
    CommitteeEvaluator,
)


class ConstantCalculator:
    def __init__(self, energy, forces):
        self.energy = energy
        self.forces = np.asarray(forces, dtype=float)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.energy

    def get_forces(self, atoms=None):
        return self.forces.copy()


class FailingCalculator:
    def get_potential_energy(self, atoms=None, force_consistent=False):
        raise RuntimeError("member failed")


def _atoms(x=0.0):
    return Atoms("H", positions=[[x, 0.0, 0.0]])


def _factories(energies=(0.0, 1.0, 2.0)):
    return [
        lambda energy=energy: ConstantCalculator(energy, [[energy, 0.0, 0.0]])
        for energy in energies
    ]


def test_committee_known_mean_and_standard_deviation():
    result = CommitteeEvaluator(_factories()).evaluate(_atoms())

    assert result.mean_energy == pytest.approx(1.0)
    assert result.energy_std == pytest.approx(np.std([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(result.mean_forces, [[1.0, 0.0, 0.0]])
    assert result.max_atom_force_std == pytest.approx(np.std([0.0, 1.0, 2.0]))
    assert result.rms_force_disagreement == pytest.approx(
        np.sqrt(np.mean(np.array([[-1.0, 0, 0], [0, 0, 0], [1, 0, 0]]) ** 2))
    )


def test_partial_member_failure_is_recorded_when_minimum_is_met():
    factories = [*_factories((0.0, 2.0)), FailingCalculator]
    result = CommitteeEvaluator(factories, minimum_valid_members=2).evaluate(_atoms())

    assert result.valid_members == 2
    assert result.failed_members == (2,)
    assert "member failed" in result.members[2].failure_reason


def test_minimum_valid_members_is_enforced():
    evaluator = CommitteeEvaluator(
        [*_factories((0.0,)), FailingCalculator, FailingCalculator],
        minimum_valid_members=2,
    )
    with pytest.raises(CommitteeEvaluationError, match="only 1"):
        evaluator.evaluate(_atoms())


def test_committee_preserves_inputs_and_uses_fresh_calculators():
    created = []

    def factory():
        calculator = ConstantCalculator(1.0, [[0.0, 0.0, 0.0]])
        created.append(calculator)
        return calculator

    atoms = _atoms(0.3)
    original = atoms.copy()
    evaluator = CommitteeEvaluator([factory, factory], minimum_valid_members=2)
    evaluator.evaluate(atoms)
    evaluator.evaluate(atoms)

    np.testing.assert_allclose(atoms.positions, original.positions)
    assert atoms.calc is None
    assert len({id(calculator) for calculator in created}) == 4


def test_committee_mean_calculator_is_ase_compatible():
    atoms = _atoms()
    atoms.calc = CommitteeCalculator(_factories(), minimum_valid_members=2)

    assert atoms.get_potential_energy() == pytest.approx(1.0)
    np.testing.assert_allclose(atoms.get_forces(), [[1.0, 0.0, 0.0]])
    assert atoms.calc.last_diagnostics.valid_members == 3


def test_path_relative_uncertainty_and_barrier_spread():
    class LinearCalculator:
        def __init__(self, scale):
            self.scale = scale

        def get_potential_energy(self, atoms=None, force_consistent=False):
            return float(atoms.positions[0, 0] * self.scale)

        def get_forces(self, atoms=None):
            return np.array([[-self.scale, 0.0, 0.0]])

    evaluator = CommitteeEvaluator(
        [lambda scale=scale: LinearCalculator(scale) for scale in (1.0, 2.0, 3.0)]
    )
    result = evaluator.evaluate_path([_atoms(0.0), _atoms(1.0), _atoms(0.5)])

    assert result.images[0].relative_energy_std == pytest.approx(0.0)
    assert result.images[1].relative_energy_std == pytest.approx(np.std([1, 2, 3]))
    assert result.member_barriers == pytest.approx((1.0, 2.0, 3.0))
    assert result.barrier_mean == pytest.approx(2.0)
    assert "proxy" in result.disclosure


def _diagnostic(index, uncertainty):
    return CommitteeEvaluator(_factories((0.0, uncertainty, 2 * uncertainty))).evaluate(
        _atoms(), index
    )


def test_candidate_selector_combines_mandatory_ranked_and_diverse_candidates():
    energies = [0.0, 0.2, 1.0, 0.8, 0.1, 0.0]
    inputs = [
        CandidateInput(i, committee=_diagnostic(i, float(i + 1)))
        for i in range(len(energies))
    ]
    inputs[4] = CandidateInput(4, committee=inputs[4].committee, failed_model=True)
    config = CandidateSelectionConfig(
        n_select=5,
        peak_neighbors=1,
        minimum_path_separation=1,
        include_endpoints_during_bootstrap=True,
    )

    result = select_active_learning_candidates(energies, inputs, config, bootstrap=True)

    assert {0, 1, 2, 3, 4, 5} <= set(result.selected_indices)
    reasons = {item.image_index: item.reasons for item in result.selected}
    assert "peak" in reasons[2]
    assert "failed_model" in reasons[4]
    assert "bootstrap_endpoint" in reasons[0]
    assert result.fallback_reason is None


def test_candidate_selector_fallback_is_explicit_and_deterministic():
    energies = [0.0, 0.4, 1.0, 0.2, 0.0]
    config = CandidateSelectionConfig(
        n_select=3,
        include_peak=False,
        diversity=False,
        include_endpoints_during_bootstrap=False,
    )

    first = select_active_learning_candidates(energies, [], config)
    second = select_active_learning_candidates(energies, [], config)

    assert first == second
    assert first.fallback_reason == "uncertainty_unavailable_ranked_by_relative_energy"
    assert first.selected_indices == (1, 2, 3)
    assert all("fallback" in item.reasons[0] for item in first.selected)


def test_candidate_selector_avoids_duplicate_indices_and_relaxes_diversity():
    result = select_active_learning_candidates(
        [0.0, 1.0, 0.9, 0.8],
        [],
        CandidateSelectionConfig(
            n_select=4,
            peak_neighbors=1,
            minimum_path_separation=2,
            include_endpoints_during_bootstrap=False,
        ),
    )

    assert len(result.selected_indices) == len(set(result.selected_indices)) == 4
    assert any("diversity_relaxed" in item.reasons for item in result.selected)
