"""RESEARCH_PLAN.md Phase 3: evaluate one leave-one-element-out fold.

For the held-out element, evaluates barrier prediction error at FIXED
geometry (the held-out element's own DFT-relaxed reference and peak images,
never touched during training or selection) for:
  - zero-shot MACE-MP-0 medium (no fine-tuning),
  - each AL-fine-tuned seed's checkpoint,
  - each random-fine-tuned seed's checkpoint,
against the true DFT barrier (REF_energy at image_index 3 minus REF_energy
at image_index 0). Evaluating all conditions at the SAME fixed geometry
(rather than re-running NEB per model) isolates model accuracy as the only
variable -- re-optimizing the path per model would confound geometric
differences with accuracy differences.

Checkpoint loading reuses the pattern already proven working in this
project's earlier (unpublished, smoke-test-only) fine-tuning validation:
mace_run_train writes a *.model file directly into --model_dir; the most
recently written one is loaded via mace.calculators.MACECalculator.

Run:
    python reproducibility/finetuning/_evaluate_fold.py <held_out>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ase.io import read
from vacancy_benchmark_suite import SYSTEMS

MATERIALS = ["al", "cu", "si", "mg", "fe"]
POOLS_DIR = Path("phase3_pools")
RUNS_DIR = Path("phase3_finetune_runs")
RESULTS_DIR = Path("phase3_results")
FOUNDATION_MODEL = "medium"
DEVICE = "cuda"
SEEDS = (1, 2, 3)


def _load_geometries(held_out: str):
    frames = read(POOLS_DIR / held_out / "held_out_full_path.extxyz", index=":")
    by_index = {f.info["image_index"]: f for f in frames}
    return (
        by_index[0].copy(),
        by_index[3].copy(),
        (by_index[3].info["REF_energy"] - by_index[0].info["REF_energy"]),
    )


def _barrier_under(calc_factory, reference, peak) -> float:
    ref = reference.copy()
    ref.calc = calc_factory()
    e_ref = float(ref.get_potential_energy())
    pk = peak.copy()
    pk.calc = calc_factory()
    e_peak = float(pk.get_potential_energy())
    return e_peak - e_ref


def _zero_shot_calc():
    from mace.calculators import mace_mp

    return mace_mp(
        model=FOUNDATION_MODEL, dispersion=False, default_dtype="float64", device=DEVICE
    )


def _finetuned_calc(held_out: str, condition: str, seed: int):
    from mace.calculators import MACECalculator

    run_dir = RUNS_DIR / held_out / condition / f"seed{seed}"
    model_files = sorted(run_dir.glob("*.model"))
    if not model_files:
        raise RuntimeError(f"no *.model checkpoint found in {run_dir}")
    model_path = model_files[-1]
    return MACECalculator(model_paths=str(model_path), device=DEVICE)


def main(held_out: str) -> None:
    if held_out not in MATERIALS:
        raise SystemExit(f"unknown element {held_out!r}, expected one of {MATERIALS}")

    reference, peak, true_barrier = _load_geometries(held_out)
    print(
        f"=== {SYSTEMS[held_out].symbol} held out; true DFT barrier "
        f"= {true_barrier:.4f} eV ==="
    )

    results: dict = {
        "held_out": held_out,
        "true_barrier_eV": true_barrier,
        "conditions": {},
    }

    zero_shot_barrier = _barrier_under(_zero_shot_calc, reference, peak)
    zero_shot_error = abs(zero_shot_barrier - true_barrier)
    print(
        f"zero-shot MACE-MP-0 medium: predicted={zero_shot_barrier:.4f} eV, "
        f"|error|={zero_shot_error:.4f} eV"
    )
    results["zero_shot"] = {
        "predicted_barrier_eV": zero_shot_barrier,
        "abs_error_eV": zero_shot_error,
    }

    for condition in ("al", "random"):
        per_seed = []
        for seed in SEEDS:
            run_dir = RUNS_DIR / held_out / condition / f"seed{seed}"
            if not run_dir.exists() or not list(run_dir.glob("*.model")):
                print(f"  {condition} seed={seed}: no checkpoint yet, skipping")
                continue
            predicted = _barrier_under(
                lambda h=held_out, c=condition, s=seed: _finetuned_calc(h, c, s),
                reference,
                peak,
            )
            error = abs(predicted - true_barrier)
            print(
                f"  {condition} seed={seed}: predicted={predicted:.4f} eV, "
                f"|error|={error:.4f} eV"
            )
            per_seed.append(
                {"seed": seed, "predicted_barrier_eV": predicted, "abs_error_eV": error}
            )
        results["conditions"][condition] = per_seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{held_out}_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python _evaluate_fold.py <held_out>")
    main(sys.argv[1].lower())
