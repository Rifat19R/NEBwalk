"""RESEARCH_PLAN.md Phase 3: run one real MACE fine-tuning job for the
leave-one-element-out AL-vs-random comparison.

Uses MACE-MP-0 "medium" uniformly as the foundation model for every
fine-tuning and zero-shot evaluation in Phase 3, regardless of which model
size (see vacancy_benchmark_suite.mace_model_for(), small by default, medium
for al/ag/ni/si) originally generated each element's NEB path geometry back
in Phase 1/2 -- Phase 3 evaluates model accuracy at FIXED geometries, so the
path-generation model choice does not bias this comparison, only which
specific images ended up as data points.

E0s are passed explicitly (not --E0s=average, known unreliable from this
project's own earlier smoke-test debugging, and not the foundation model's
built-in E0s, which use a different DFT reference level -- see
nebwalk.finetune.TRAINING_SET_DISCLOSURE), built from the already-validated
datasets/<Symbol>/isolated_atom_reference.json for the 4 training elements
(the held-out element's atoms never appear in the training pool at all, so
it needs no E0 entry here).

Run:
    python reproducibility/finetuning/_run_finetune_fold.py <held_out> <al|random> <seed>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ase.data import atomic_numbers
from vacancy_benchmark_suite import SYSTEMS

from nebwalk.finetune import load_isolated_atom_reference

MATERIALS = ["al", "cu", "si", "mg", "fe"]
POOLS_DIR = Path("phase3_pools")
RUNS_DIR = Path("phase3_finetune_runs")
FOUNDATION_MODEL = "medium"
DEVICE = "cuda"
MAX_NUM_EPOCHS = 100
PATIENCE = 20
VALID_FRACTION = 0.2


def e0s_for(held_out: str) -> dict[int, float]:
    train_elements = [m for m in MATERIALS if m != held_out]
    e0s: dict[int, float] = {}
    for material in train_elements:
        symbol = SYSTEMS[material].symbol
        ref = load_isolated_atom_reference(
            f"datasets/{symbol}/isolated_atom_reference.json"
        )
        e0s[atomic_numbers[symbol]] = ref.energy_eV
    return e0s


def main(held_out: str, condition: str, seed: int) -> None:
    if held_out not in MATERIALS:
        raise SystemExit(f"unknown element {held_out!r}, expected one of {MATERIALS}")
    if condition not in ("al", "random"):
        raise SystemExit("condition must be 'al' or 'random'")

    pool_path = POOLS_DIR / held_out / f"{condition}_pool.extxyz"
    if not pool_path.exists():
        raise SystemExit(
            f"missing {pool_path}; run reproducibility/finetuning/_build_leave_one_out_pools.py first"
        )

    e0s = e0s_for(held_out)
    name = f"{held_out}_holdout_{condition}_seed{seed}"
    run_dir = RUNS_DIR / held_out / condition / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "mace_run_train",
        f"--name={name}",
        f"--foundation_model={FOUNDATION_MODEL}",
        f"--train_file={pool_path.resolve()}",
        f"--valid_fraction={VALID_FRACTION}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        f"--E0s={json.dumps(e0s)}",
        "--multiheads_finetuning=False",
        f"--device={DEVICE}",
        f"--max_num_epochs={MAX_NUM_EPOCHS}",
        f"--patience={PATIENCE}",
        "--batch_size=1",
        f"--seed={seed}",
        f"--model_dir={run_dir.resolve()}",
        f"--log_dir={run_dir.resolve()}",
        f"--checkpoints_dir={run_dir.resolve()}",
        f"--results_dir={run_dir.resolve()}",
    ]
    print("Running:", " ".join(command))
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    print(f"exit code: {result.returncode}; log: {log_path}")
    if result.returncode != 0:
        raise SystemExit(f"mace_run_train failed for {name}; see {log_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python _run_finetune_fold.py <held_out> <al|random> <seed>"
        )
    main(sys.argv[1].lower(), sys.argv[2].lower(), int(sys.argv[3]))
