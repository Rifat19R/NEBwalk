"""Enforce the module dependency direction in docs/reference/architecture.md.

The NEB core must stay usable without, and independent of, the MLIP/DFT
handoff and generic ML-lifecycle layers, so that the latter can later be
relocated behind compatibility shims without touching NEB physics.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "NEBwalk"

NEB_CORE = (
    "engine",
    "forces",
    "interpolate",
    "neb",
    "optimize",
    "output",
    "qe",
    "recovery",
    "reproduce",
)
HANDOFF = (
    "active",
    "campaign",
    "candidate_selection",
    "cli",
    "dryrun",
    "finetune",
    "label",
    "labeling",
    "selection",
    "uncertainty",
    "validate",
)
LIFECYCLE = ("campaign_state", "datasets", "mlip", "reporting")
OUTER_LAYERS = HANDOFF + LIFECYCLE


def _internal_imports(path: Path) -> set[str]:
    """Top-level NEBwalk modules/subpackages imported by ``path``."""
    package = ["NEBwalk", *path.relative_to(PACKAGE).parent.parts]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        targets: list[list[str]] = []
        if isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = base + (node.module.split(".") if node.module else [])
            if node.module:
                targets.append(module)
            else:
                targets.extend(module + [alias.name] for alias in node.names)
        elif isinstance(node, ast.Import):
            targets.extend(alias.name.split(".") for alias in node.names)
        for target in targets:
            if target[0] == "NEBwalk" and len(target) > 1:
                imported.add(target[1])
    return imported


def _lifecycle_files() -> list[Path]:
    files = [PACKAGE / f"{name}.py" for name in LIFECYCLE if name != "mlip"]
    return files + sorted((PACKAGE / "mlip").glob("*.py"))


def test_layers_cover_every_module():
    modules = {
        path.stem for path in PACKAGE.glob("*.py") if path.stem != "__init__"
    } | {"mlip"}
    assert modules == set(NEB_CORE) | set(OUTER_LAYERS)


@pytest.mark.parametrize("module", NEB_CORE)
def test_neb_core_does_not_import_outer_layers(module):
    leaked = _internal_imports(PACKAGE / f"{module}.py") & set(OUTER_LAYERS)
    assert leaked == set()


@pytest.mark.parametrize("path", _lifecycle_files(), ids=lambda path: path.stem)
def test_lifecycle_layer_does_not_import_neb_or_handoff_code(path):
    leaked = _internal_imports(path) & (set(NEB_CORE) | set(HANDOFF))
    assert leaked == set()
