"""RESEARCH_PLAN.md Phase 3: build leave-one-element-out AL/random training
pools from the 5-element publication subset's canonical datasets.

For each of 5 folds (one element held out), builds two training pools from
the OTHER 4 elements' already-validated datasets/<Symbol>/<material>_train.
extxyz files:
  - AL pool: each element's reference + peak_plus_neighbors frames (the
    active-selection strategy's actual choice).
  - Random pool: each element's reference + an equal-sized (3 frames)
    uniform-random draw from the same element's non-reference candidates,
    with a deterministic (SHA-256-derived, not Python's randomized
    hash()) seed per (fold, element) pair, logged so the exact draw is
    reproducible and auditable.

Both pools are written with plain ase.io.write(), not
NEBwalk.datasets.write_dataset() -- a genuine multi-element pool has
different dft_settings_hash per element by construction (different
pseudopotentials/cutoffs), which write_dataset()'s validator correctly
refuses to mix into one canonical dataset. This mirrors the existing,
documented precedent in NEBwalk.finetune.combine_training_sets() for the
same reason. Each source file was already individually validated through
NEBwalk.datasets.load_dataset() before being combined here.

The held-out element's own full 5-frame path is copied alongside as the
evaluation ground truth (its DFT barrier = REF_energy at image_index 3 minus
REF_energy at image_index 0, recomputed here as a cross-check against the
DFTLabel-derived value already in its validation_summary.json).

Run:
    python reproducibility/finetuning/_build_leave_one_out_pools.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ase.io import write
from vacancy_benchmark_suite import SYSTEMS

from NEBwalk.datasets import load_dataset

MATERIALS = ["al", "cu", "si", "mg", "fe"]
N_RANDOM_DRAW = 3  # matches peak_plus_neighbors' non-reference frame count
OUTPUT_DIR = Path("phase3_pools")


def _deterministic_seed(*parts: str) -> int:
    """SHA-256-derived seed -- reproducible across processes/machines,
    unlike Python's hash() (randomized per-process via PYTHONHASHSEED)."""
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_element_frames(material: str):
    path = Path(f"datasets/{SYSTEMS[material].symbol}/{material}_train.extxyz")
    return load_dataset(path)


def _al_selected(frames):
    return [
        f
        for f in frames
        if f.info["selection_reason"] in ("reference", "peak_plus_neighbors")
    ]


def _random_selected(frames, seed: int):
    reference = [f for f in frames if f.info["selection_reason"] == "reference"]
    candidates = [f for f in frames if f.info["selection_reason"] != "reference"]
    if len(candidates) < N_RANDOM_DRAW:
        raise RuntimeError(
            f"only {len(candidates)} non-reference candidates, need {N_RANDOM_DRAW}"
        )
    rng = random.Random(seed)
    chosen = rng.sample(candidates, k=N_RANDOM_DRAW)
    return reference + chosen


def build_fold(held_out: str) -> Path:
    train_elements = [m for m in MATERIALS if m != held_out]
    al_frames = []
    random_frames = []
    fold_log: dict = {"held_out": held_out, "random_draws": {}}

    for material in train_elements:
        frames = _load_element_frames(material)
        al_subset = _al_selected(frames)
        seed = _deterministic_seed(held_out, material)
        random_subset = _random_selected(frames, seed)

        fold_log["random_draws"][material] = {
            "seed": seed,
            "al_indices": sorted(f.info["image_index"] for f in al_subset),
            "random_indices": sorted(f.info["image_index"] for f in random_subset),
        }
        al_frames.extend(al_subset)
        random_frames.extend(random_subset)

    held_out_frames = _load_element_frames(held_out)
    ref_energy = next(
        f.info["REF_energy"] for f in held_out_frames if f.info["image_index"] == 0
    )
    peak_energy = next(
        f.info["REF_energy"] for f in held_out_frames if f.info["image_index"] == 3
    )
    fold_log["held_out_true_barrier_eV"] = peak_energy - ref_energy

    out = OUTPUT_DIR / held_out
    out.mkdir(parents=True, exist_ok=True)
    write(out / "al_pool.extxyz", al_frames, format="extxyz")
    write(out / "random_pool.extxyz", random_frames, format="extxyz")
    write(out / "held_out_full_path.extxyz", held_out_frames, format="extxyz")
    (out / "fold_log.json").write_text(json.dumps(fold_log, indent=2) + "\n")

    print(
        f"fold={held_out}: AL={len(al_frames)} frames, "
        f"random={len(random_frames)} frames, "
        f"held-out true barrier={fold_log['held_out_true_barrier_eV']:.4f} eV"
    )
    return out


def main() -> None:
    for held_out in MATERIALS:
        build_fold(held_out)


if __name__ == "__main__":
    main()
