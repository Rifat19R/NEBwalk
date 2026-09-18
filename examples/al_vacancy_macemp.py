"""
Example 4: Al vacancy diffusion in bulk FCC Al using MACE-MP-0

A single Al vacancy hops to an adjacent lattice site via a nearest-neighbour
jump in the (111) plane. This is a classic point-defect NEB benchmark.

Why this system:
  - MACE-MP-0 was trained on Materials Project bulk structures → accurate here.
  - 3D periodic boundary conditions (pbc=True in all directions).
  - Only one line changes vs any other calculator: make_calc().
  - Directly relevant to diffusion and defect physics in real materials.

Expected barriers:
  DFT (PBE)       : ~0.61 eV  (Mantina et al., Phys. Rev. Lett. 100, 2008)
  Experimental    : ~0.60–0.68 eV
  MACE-MP-0 small : ~0.55–0.70 eV  (bulk Al is in training distribution)

System:
  2×2×2 FCC Al supercell = 32 atoms.  One vacancy created by removing one Al
  at the corner site.  The adjacent face-centre Al (2.86 Å away) jumps into
  the vacancy.  The supercell is 8.1 × 8.1 × 8.1 Å — large enough that the
  vacancy does not interact with its periodic images through MACE-MP-0's 6 Å
  cutoff.

Run:
    python examples/al_vacancy_macemp.py
"""

import logging
import warnings

import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from mace.calculators import mace_mp

from NEBwalk import NEB, linear_interpolate

logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "small"  # "medium" for higher accuracy (~20 MB extra download)
DEVICE = "cpu"
DTYPE = "float64"  # mandatory for geometry optimisation with MACE-MP-0

N_IMAGES = 5
K_SPRING = 0.10  # eV/Å²
FMAX_RELAX = 0.01  # eV/Å
FMAX_NEB = 0.05  # eV/Å

print("System          : Al vacancy diffusion in bulk FCC Al")
print(f"MACE-MP-0 model : {MODEL}")
print(f"Device          : {DEVICE}")
print(f"dtype           : {DTYPE}\n")

# Pre-load model (triggers the one-time download message)
_ = mace_mp(model=MODEL, device=DEVICE, default_dtype=DTYPE)


def make_calc():
    return mace_mp(model=MODEL, device=DEVICE, default_dtype=DTYPE)


# ---------------------------------------------------------------------------
# Build 31-atom system with one vacancy
#
# Full 32-atom 2×2×2 FCC Al supercell (cubic, pbc=True):
#   Atom 0: [0.000, 0.000, 0.000]  ← vacancy site
#   Atom 1: [2.025, 2.025, 0.000]  ← jumping atom (nearest neighbour)
#   Atom 2: [2.025, 0.000, 2.025]  ← etc.
#
# After del[0]: the 31-atom system has atom 0 (formerly atom 1) as the
# jumping atom and an implicit vacancy at [0, 0, 0].
# ---------------------------------------------------------------------------
al_super = bulk("Al", "fcc", a=4.05, cubic=True).repeat([2, 2, 2])  # 32 atoms
vac_pos = al_super.positions[0].copy()  # [0.000, 0.000, 0.000]
jump_pos = al_super.positions[1].copy()  # [2.025, 2.025, 0.000]

print(f"Cell            : {al_super.cell[0, 0]:.3f} Å (cubic)")
print(f"Vacancy site    : {vac_pos}")
print(f"Jumping atom    : {jump_pos}")
print(
    f"Jump distance   : {np.linalg.norm(jump_pos - vac_pos):.3f} Å  "
    f"(= a/√2, FCC nearest-neighbour)\n"
)

# Create 31-atom base system (vacancy at former site 0)
al_31 = al_super.copy()
del al_31[0]

# ---------------------------------------------------------------------------
# Relax endpoints
# ---------------------------------------------------------------------------
print("Relaxing initial state  (jumping atom at NN site, vacancy at [0,0,0]) ...")
initial = al_31.copy()
initial.calc = make_calc()
BFGS(initial, logfile=None).run(fmax=FMAX_RELAX)
E0 = initial.get_potential_energy()
print(f"  E_initial = {E0:.4f} eV")

print("Relaxing final state    (jumping atom at [0,0,0], vacancy at NN site) ...")
final = al_31.copy()
final.positions[0] = vac_pos  # move jumping atom into the vacancy
final.calc = make_calc()
BFGS(final, logfile=None).run(fmax=FMAX_RELAX)
E1 = final.get_potential_energy()
print(f"  E_final   = {E1:.4f} eV")
print(f"  ΔE        = {E1 - E0:.5f} eV  (should be ~0 by symmetry)\n")

# ---------------------------------------------------------------------------
# NEB
# linear_interpolate uses MIC: jump displacement 2.86 Å < 4.05 Å (half cell)
# so no wrapping correction needed, but it is handled correctly regardless.
# ---------------------------------------------------------------------------
print(f"Building NEB path with {N_IMAGES} intermediate images ...")
images = linear_interpolate(initial, final, n_images=N_IMAGES)
for img in images:
    img.calc = make_calc()

neb = NEB(images, k=K_SPRING, climb=True, climb_delay=50)
converged = neb.optimize(fmax=FMAX_NEB, max_steps=300)

print(f"\nConverged          : {converged}")
print(f"Forward barrier    : {neb.get_barrier():.4f} eV")
print("DFT (PBE) ref.     : ~0.61 eV  (Mantina et al., PRL 100, 215901, 2008)")
print("Experimental       : ~0.60–0.68 eV")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
neb.plot(
    "al_vacancy_macemp_profile.png",
    title="Al vacancy diffusion in FCC Al [MACE-MP-0, CI-NEB]",
)
neb.save_csv("al_vacancy_macemp_profile.csv")
neb.save_trajectory("al_vacancy_macemp.traj")
print("\nDone.  View trajectory:  ase gui al_vacancy_macemp.traj")
