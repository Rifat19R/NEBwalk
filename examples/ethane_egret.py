"""
Example 3: Ethane C–C torsional barrier using Egret-1t

Reaction : staggered C2H6 → [eclipsed TS] → staggered C2H6  (120° CH3 rotation)
Expected barrier : ~0.126 eV  (12.2 kJ/mol, experimental)

Demonstrates:
  - idpp_interpolate for torsional paths (avoids atomic clashes)
  - Variable spring constants (k_min = k/3)
  - Parallel image evaluation (n_workers = N_IMAGES)

Note: an explicit 20 Å cubic cell is required for MACE neighbour lists.
      Without it, matscipy raises "Failed to allocate seed array".

Run:
    python examples/ethane_egret.py
"""

import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import MACECalculator

from NEBwalk import NEB, idpp_interpolate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "EGRET_1T.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = 7  # intermediate images
K_MAX = 0.10  # eV/Å² — spring constant at saddle point
K_MIN = 0.033  # eV/Å² — spring constant far from saddle (k/3)

# ---------------------------------------------------------------------------
# Geometry parameters (D3d ethane)
# ---------------------------------------------------------------------------
CC = 0.770
r_CH = 1.090
_dz = abs(r_CH * np.cos(np.radians(111.2)))
_rl = r_CH * np.sin(np.radians(111.2))
CELL = np.diag([20.0, 20.0, 20.0])


def ethane(phi_C2_deg: float) -> Atoms:
    """
    C2H6 Atoms at given C2 methyl dihedral angle (degrees).
    phi = 60°  → staggered; phi = 180° → eclipsed.
    """
    phi = np.radians(phi_C2_deg)
    pos = [
        [0.0, 0.0, -CC],
        [0.0, 0.0, +CC],
    ]
    for k in range(3):
        a = k * (2 * np.pi / 3)
        pos.append([_rl * np.cos(a), _rl * np.sin(a), -CC - _dz])
    for k in range(3):
        a = phi + k * (2 * np.pi / 3)
        pos.append([_rl * np.cos(a), _rl * np.sin(a), +CC + _dz])
    atoms = Atoms("C2H6", positions=pos)
    atoms.set_cell(CELL)
    atoms.pbc = False
    atoms.center()
    return atoms


def make_calc():
    kw = dict(device=DEVICE, default_dtype="float32")
    try:
        return MACECalculator(model_paths=MODEL_PATH, **kw)
    except TypeError:
        return MACECalculator(model_path=MODEL_PATH, **kw)


# ---------------------------------------------------------------------------
# Relax endpoints
# ---------------------------------------------------------------------------
print(f"Device   : {DEVICE}")
print(f"Model    : {MODEL_PATH}\n")

print("Relaxing initial state (staggered, phi = 60°) ...")
initial = ethane(60.0)
initial.calc = make_calc()
BFGS(initial, logfile=None).run(fmax=0.01)
E0 = initial.get_potential_energy()
print(f"  E_initial = {E0:.4f} eV")

print("Relaxing final state   (staggered, phi = 180°) ...")
final = ethane(180.0)
final.calc = make_calc()
BFGS(final, logfile=None).run(fmax=0.01)
E1 = final.get_potential_energy()
print(f"  E_final   = {E1:.4f} eV")
print(f"  ΔE        = {E1 - E0:.5f} eV  (should be ~0 by symmetry)")

# ---------------------------------------------------------------------------
# IDPP interpolation
#
# Linear Cartesian interpolation moves atoms through each other for torsional
# paths. IDPP (Smidstrup et al. 2014) preserves pairwise distance structure,
# producing a chemically sensible initial path with no steric clashes.
# ---------------------------------------------------------------------------
print(f"\nBuilding {N_IMAGES} intermediate images with IDPP interpolation ...")
images = idpp_interpolate(initial, final, n_images=N_IMAGES)
for img in images:
    img.calc = make_calc()

# ---------------------------------------------------------------------------
# NEB with variable spring constants and parallel evaluation
# ---------------------------------------------------------------------------
neb = NEB(
    images,
    k=K_MAX,
    k_min=K_MIN,  # variable springs: concentrate images near TS
    climb=True,
    climb_delay=60,
    n_workers=N_IMAGES,  # one thread per image — Egret-1t releases the GIL
)
converged = neb.optimize(fmax=0.05, max_steps=400)

print(f"\nConverged       : {converged}")
print(f"Forward barrier : {neb.get_barrier():.4f} eV")
print("Literature      : ~0.126 eV  (12.2 kJ/mol)")
print(f"k_springs       : {neb.get_spring_constants().round(4)}")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
neb.plot(
    "ethane_egret_profile.png", title="Ethane C–C torsion [Egret-1t, CI-NEB, IDPP]"
)
neb.save_csv("ethane_egret_profile.csv")
neb.save_trajectory("ethane_egret.traj")

print("\nDone.  View trajectory with:  ase gui ethane_egret.traj")
