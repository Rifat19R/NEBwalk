from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nebwalk.qe import (
    QEParams,
    _parse_qe_energy_forces,
    make_qe_factory,
    validate_qe_setup,
)


def write_pseudos(tmp_path: Path, *names: str) -> Path:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    for name in names:
        (pseudo_dir / name).write_text("pseudo", encoding="utf-8")
    return pseudo_dir


@pytest.fixture(autouse=True)
def _reset_qe_globals():
    """Reset nebwalk.qe module globals between tests to ensure isolation.

    test_factory_rejects_unknown_magnetization_species has no Espresso mock,
    so it triggers the real ASE import and sets module-level EspressoProfile.
    Without this fixture, subsequent tests inherit that state and break.
    """
    import nebwalk.qe as _qe

    orig_espresso = _qe.Espresso
    orig_profile = _qe.EspressoProfile
    yield
    _qe.Espresso = orig_espresso
    _qe.EspressoProfile = orig_profile


def test_qe_params_defaults_are_neb_safe() -> None:
    params = QEParams()

    assert params.ecutwfc == 40.0
    assert params.ecutrho == 320.0
    assert params.kpts == (1, 1, 1)
    assert params.koffset == (0, 0, 0)
    assert params.occupations == "smearing"
    assert params.smearing == "marzari-vanderbilt"
    assert params.degauss == 0.02
    assert params.conv_thr == 1.0e-8
    assert params.mixing_beta == 0.3
    assert params.nspin == 1


def test_validate_qe_setup_accepts_existing_inputs(tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")

    with patch("nebwalk.qe.shutil.which", return_value="/usr/bin/pw.x"):
        validate_qe_setup(pseudo_dir, {"Al": "Al.UPF"}, command="pw.x")


def test_validate_qe_setup_rejects_missing_pseudo_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="directory"):
        validate_qe_setup(tmp_path / "missing", {"Al": "Al.UPF"})


def test_validate_qe_setup_rejects_empty_pseudo_map(tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path)

    with pytest.raises(ValueError, match="At least one"):
        validate_qe_setup(pseudo_dir, {})


def test_validate_qe_setup_rejects_missing_pseudo_file(tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")

    with pytest.raises(FileNotFoundError, match="Missing QE pseudopotential"):
        validate_qe_setup(pseudo_dir, {"Mg": "Mg.UPF"})


def test_validate_qe_setup_rejects_missing_binary(tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")

    with patch("nebwalk.qe.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="executable"):
            validate_qe_setup(pseudo_dir, {"Al": "Al.UPF"}, command="missing_pw.x")


def test_validate_qe_setup_checks_mpi_launcher_and_pw(tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    seen: list[str] = []

    def fake_which(binary: str) -> str:
        seen.append(binary)
        return f"/usr/bin/{binary}"

    with patch("nebwalk.qe.shutil.which", side_effect=fake_which):
        validate_qe_setup(
            pseudo_dir,
            {"Al": "Al.UPF"},
            command="mpirun -np 4 pw.x",
        )

    assert seen == ["mpirun", "pw.x"]


@patch("nebwalk.qe.Espresso", autospec=True)
def test_make_qe_factory_returns_callable(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")

    factory = make_qe_factory(
        QEParams(),
        pseudo_dir,
        {"Al": "Al.UPF"},
        base_dir=tmp_path / "qe",
    )

    assert callable(factory)
    mock_espresso.assert_not_called()


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_creates_unique_dirs(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    factory = make_qe_factory(
        QEParams(),
        pseudo_dir,
        {"Al": "Al.UPF"},
        base_dir=tmp_path / "qe",
    )

    factory()
    factory()

    first_kwargs = mock_espresso.call_args_list[0].kwargs
    second_kwargs = mock_espresso.call_args_list[1].kwargs
    assert first_kwargs["directory"].endswith("image_000")
    assert second_kwargs["directory"].endswith("image_001")
    assert (tmp_path / "qe" / "image_000" / "tmp").is_dir()
    assert (tmp_path / "qe" / "image_001" / "tmp").is_dir()


@patch("nebwalk.qe.Espresso", autospec=True)
def test_counter_resets_per_factory(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    first = make_qe_factory(
        QEParams(),
        pseudo_dir,
        {"Al": "Al.UPF"},
        base_dir=tmp_path / "qe",
    )
    second = make_qe_factory(
        QEParams(),
        pseudo_dir,
        {"Al": "Al.UPF"},
        base_dir=tmp_path / "qe",
    )

    first()
    second()

    assert mock_espresso.call_args_list[0].kwargs["directory"].endswith("image_000")
    assert mock_espresso.call_args_list[1].kwargs["directory"].endswith("image_000")


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_uses_qe_control_defaults(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    factory = make_qe_factory(
        QEParams(),
        pseudo_dir,
        {"Al": "Al.UPF"},
        base_dir=tmp_path / "qe",
    )

    factory()

    control = mock_espresso.call_args.kwargs["input_data"]["control"]
    assert control["calculation"] == "scf"
    assert control["restart_mode"] == "from_scratch"
    assert control["tprnfor"] is True
    assert control["tstress"] is False
    assert control["disk_io"] == "low"
    assert Path(control["pseudo_dir"]) == pseudo_dir.resolve()
    assert Path(control["outdir"]) == (tmp_path / "qe" / "image_000" / "tmp").resolve()


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_uses_system_and_electron_params(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    params = QEParams(
        ecutwfc=60.0,
        ecutrho=480.0,
        kpts=(3, 3, 1),
        koffset=(1, 1, 0),
        conv_thr=1.0e-10,
        mixing_beta=0.2,
    )
    factory = make_qe_factory(params, pseudo_dir, {"Al": "Al.UPF"})

    factory()

    kwargs = mock_espresso.call_args.kwargs
    system = kwargs["input_data"]["system"]
    electrons = kwargs["input_data"]["electrons"]
    assert system["ecutwfc"] == 60.0
    assert system["ecutrho"] == 480.0
    assert system["smearing"] == "marzari-vanderbilt"
    assert system["degauss"] == 0.02
    assert electrons["conv_thr"] == 1.0e-10
    assert electrons["mixing_beta"] == 0.2
    assert kwargs["kpts"] == (3, 3, 1)
    assert kwargs["koffset"] == (1, 1, 0)


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_omits_unused_smearing(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    factory = make_qe_factory(
        QEParams(occupations="fixed"),
        pseudo_dir,
        {"Al": "Al.UPF"},
    )

    factory()

    system = mock_espresso.call_args.kwargs["input_data"]["system"]
    assert system["occupations"] == "fixed"
    assert "smearing" not in system
    assert "degauss" not in system


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_maps_starting_magnetization(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Fe.UPF", "O.UPF")
    params = QEParams(nspin=2, starting_magnetization={"Fe": 0.6, "O": 0.1})
    factory = make_qe_factory(params, pseudo_dir, {"Fe": "Fe.UPF", "O": "O.UPF"})

    factory()

    system = mock_espresso.call_args.kwargs["input_data"]["system"]
    assert system["nspin"] == 2
    assert system["starting_magnetization(1)"] == 0.6
    assert system["starting_magnetization(2)"] == 0.1


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_accepts_integer_magnetization_indices(
    mock_espresso,
    tmp_path: Path,
) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Fe.UPF")
    params = QEParams(nspin=2, starting_magnetization={1: 0.7})
    factory = make_qe_factory(params, pseudo_dir, {"Fe": "Fe.UPF"})

    factory()

    system = mock_espresso.call_args.kwargs["input_data"]["system"]
    assert system["starting_magnetization(1)"] == 0.7


@patch("nebwalk.qe.Espresso")
def test_factory_rejects_unknown_magnetization_species(
    mock_espresso,
    tmp_path: Path,
) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Fe.UPF")
    params = QEParams(nspin=2, starting_magnetization={"Co": 0.7})
    factory = make_qe_factory(params, pseudo_dir, {"Fe": "Fe.UPF"})

    with pytest.raises(ValueError, match="not in pseudos"):
        factory()


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_passes_extra_input_sections(mock_espresso, tmp_path: Path) -> None:
    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    params = QEParams(
        extra_control={"verbosity": "high"},
        extra_system={"input_dft": "PBE"},
        extra_electrons={"electron_maxstep": 200},
    )
    factory = make_qe_factory(params, pseudo_dir, {"Al": "Al.UPF"})

    factory()

    input_data = mock_espresso.call_args.kwargs["input_data"]
    assert input_data["control"]["verbosity"] == "high"
    assert input_data["system"]["input_dft"] == "PBE"
    assert input_data["electrons"]["electron_maxstep"] == 200


@patch("nebwalk.qe.Espresso", autospec=True)
def test_factory_uses_espresso_profile_when_available(
    mock_espresso,
    tmp_path: Path,
) -> None:
    class FakeEspressoProfile:
        def __init__(self, command: str, pseudo_dir: str) -> None:
            self.command = command
            self.pseudo_dir = pseudo_dir

    pseudo_dir = write_pseudos(tmp_path, "Al.UPF")
    with patch("nebwalk.qe.EspressoProfile", FakeEspressoProfile):
        factory = make_qe_factory(
            QEParams(kpts=(2, 2, 2)),
            pseudo_dir,
            {"Al": "Al.UPF"},
            base_dir=tmp_path / "qe",
            command="mpirun -np 4 pw.x",
        )

        factory()

    kwargs = mock_espresso.call_args.kwargs
    assert kwargs["pseudopotentials"] == {"Al": "Al.UPF"}
    assert kwargs["kpts"] == (2, 2, 2)
    assert kwargs["directory"].endswith("image_000")
    assert "label" not in kwargs
    assert "command" not in kwargs
    assert kwargs["profile"].command == "mpirun -np 4 pw.x"
    assert kwargs["profile"].pseudo_dir == str(pseudo_dir.resolve())


def test_parse_completed_qe_output_with_repeated_band_blocks() -> None:
    raw_output = """
     total energy              =      -10.00000000 Ry
     End of self-consistent calculation
          k = 0.0000 0.0000 0.0000 ( 10 PWs)   bands (ev):
   -1.0 -0.5
     End of self-consistent calculation
          k = 0.0000 0.0000 0.0000 ( 10 PWs)   bands (ev):
   -1.0 -0.5
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.10000000    0.00000000   -0.20000000
     atom    2 type  1   force =     0.00000000    0.30000000    0.00000000
     Total force =     0.374166     Total SCF correction =     0.000001
   JOB DONE.
"""

    energy, forces = _parse_qe_energy_forces(raw_output)

    assert energy == pytest.approx(-136.05693122)
    assert forces.shape == (2, 3)
    assert forces[0, 0] == pytest.approx(2.57110311)
    assert forces[0, 2] == pytest.approx(-5.14220622)
    assert forces[1, 1] == pytest.approx(7.71330933)
