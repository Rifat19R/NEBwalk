# benchmarks/bench_parallel_mace.py
#
# Benchmark: sequential vs parallel image evaluation
# Calculator: Egret-1t (MACE-OFF)
# System: ethane C-C torsion, 7 intermediate images
#
# Requirements:
#   - EGRET_1T.model in current directory (download from rowansci.com)
#   - pip install NEBwalk[mace]
#
# Usage:
#   cd benchmarks/
#   python bench_parallel_mace.py

import time

import numpy as np
import torch
from ase.build import molecule
from mace.calculators import mace_off

from NEBwalk import NEB, idpp_interpolate

MODEL = "EGRET_1T.model"
N_IMAGES = 7
N_REPEATS = 10
N_WARMUP = 3


def make_images():
    """Build fresh images with independent calculator instances."""
    start = molecule("C2H6")
    end = molecule("C2H6")
    end.rotate(60, "z", center=end.positions[0])
    images = idpp_interpolate(start, end, N_IMAGES)
    # Each image gets its own calculator — required for correctness
    for img in images:
        img.calc = mace_off(model=MODEL, device="cpu")
    return images


def bench(n_workers, n_threads=None):
    """Time n_workers evaluation, rebuilding images each repeat."""
    if n_threads is not None:
        torch.set_num_threads(n_threads)
    times = []
    for rep in range(N_WARMUP + N_REPEATS):
        images = make_images()
        NEB(images, k=0.1, n_workers=n_workers)
        t0 = time.perf_counter()
        # Single force evaluation step (not full optimization)
        for img in images[1:-1]:
            img.get_forces()
        elapsed = time.perf_counter() - t0
        if rep >= N_WARMUP:
            times.append(elapsed * 1000)
    return np.mean(times), np.std(times)


print(f"CPU cores visible: {torch.get_num_threads()} PyTorch threads")
print(f"System: C2H6 ethane torsion, {N_IMAGES} intermediate images\n")

print("=== Sequential (n_workers=1) ===")
seq_ms, seq_std = bench(n_workers=1)
print(f"  Mean: {seq_ms:.1f} ± {seq_std:.1f} ms")

print("\n=== Parallel (n_workers=7, default threads) ===")
par_ms, par_std = bench(n_workers=N_IMAGES)
print(f"  Mean: {par_ms:.1f} ± {par_std:.1f} ms")
print(f"  Speedup: {seq_ms / par_ms:.2f}×")

print("\n=== Parallel (n_workers=7, 1 thread/worker) ===")
par1_ms, par1_std = bench(n_workers=N_IMAGES, n_threads=1)
print(f"  Mean: {par1_ms:.1f} ± {par1_std:.1f} ms")
print(f"  Speedup: {seq_ms / par1_ms:.2f}×")
