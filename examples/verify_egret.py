"""
Sanity check: load Egret-1t and verify energy + forces on staggered ethane.
Run this BEFORE the full NEB example.
"""

import sys

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

MODEL = "EGRET_1T.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {DEVICE}")

# --- Load calculator (handle old model_path vs new model_paths API) ---
try:
    calc = MACECalculator(model_paths=MODEL, device=DEVICE, default_dtype="float32")
    print("Loaded via model_paths= (mace-torch >= 0.3.4)")
except TypeError:
    calc = MACECalculator(model_path=MODEL, device=DEVICE, default_dtype="float32")
    print("Loaded via model_path= (mace-torch < 0.3.4)")

print("Model loaded OK\n")

# --- Staggered ethane (approximate geometry; full relaxation happens in NEB script) ---
atoms = Atoms(
    "C2H6",
    positions=[
        [0.000, 0.000, -0.770],  # C1
        [0.000, 0.000, 0.770],  # C2
        [1.016, 0.000, -1.164],  # H1 (C1 methyl)
        [-0.508, 0.880, -1.164],  # H2
        [-0.508, -0.880, -1.164],  # H3
        [0.508, 0.880, 1.164],  # H4 (C2 methyl, staggered)
        [-1.016, 0.000, 1.164],  # H5
        [0.508, -0.880, 1.164],  # H6
    ],
)
atoms.calc = calc

E = atoms.get_potential_energy()
F = atoms.get_forces()

print(f"Energy       : {E:.4f} eV  (absolute; only differences matter for NEB)")
print(f"Forces shape : {F.shape}  (should be (8, 3))")
print(
    f"Max |F|      : {np.max(np.abs(F)):.4f} eV/Å (nonzero expected; approx geometry)"
)

if F.shape == (8, 3):
    print("\nAll checks passed. Ready to run examples/ethane_egret.py")
else:
    print(f"\nERROR: unexpected force shape {F.shape}. Check model file.")
    sys.exit(1)
