"""
Example 2: Al adatom diffusion on Al(100)  hollow → bridge → hollow

This is the canonical EMT/NEB benchmark used throughout the ASE documentation.
EMT gives a barrier of ~0.24 eV for this process.

Demonstrates:
  - linear_interpolate (appropriate for surface translations — no torsion)
  - Variable spring constants (k_min = k/3)
  - FixAtoms constraints on lower slab layers

Run:
    python examples/al_diffusion_emt.py

Requires: ase (pip install ase)
"""

import numpy as np
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from NEBwalk import NEB, linear_interpolate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_IMAGES = 5  # intermediate images
K_MAX = 0.10  # eV/Å² — spring constant at saddle point
K_MIN = 0.033  # eV/Å² — spring constant far from saddle (k/3)


# ---------------------------------------------------------------------------
# Build slab and relax endpoints
# ---------------------------------------------------------------------------


def make_slab():
    """Al(100) slab, 2×2×3, with Al adatom at hollow site."""
    slab = fcc100("Al", size=(2, 2, 3), vacuum=10.0)
    add_adsorbate(slab, "Al", height=1.7, position="hollow")
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    slab.calc = EMT()
    return slab


print("Building and relaxing initial state (hollow site A) ...")
initial = make_slab()
BFGS(initial, logfile=None).run(fmax=0.01)
e_initial = initial.get_potential_energy()
print(f"  E_initial = {e_initial:.4f} eV")

# Shift adatom to adjacent hollow site
final = initial.copy()
final.calc = EMT()
adatom_idx = len(final) - 1
final.positions[adatom_idx] += np.array([4.05 / 2, 0.0, 0.0])

print("Relaxing final state (hollow site B) ...")
BFGS(final, logfile=None).run(fmax=0.01)
e_final = final.get_potential_energy()
print(f"  E_final   = {e_final:.4f} eV")

# ---------------------------------------------------------------------------
# NEB with variable spring constants
# ---------------------------------------------------------------------------
print(f"\nBuilding NEB path with {N_IMAGES} intermediate images ...")
images = linear_interpolate(initial, final, n_images=N_IMAGES)
for img in images:
    img.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in img]))
    img.calc = EMT()

neb = NEB(
    images,
    k=K_MAX,
    k_min=K_MIN,  # variable springs: concentrate images near TS
    climb=True,
    climb_delay=50,
)
converged = neb.optimize(fmax=0.05, max_steps=300)

print(f"\nConverged       : {converged}")
print(f"Forward barrier : {neb.get_barrier():.4f} eV")
print(f"k_springs       : {neb.get_spring_constants().round(4)}")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
neb.plot(
    "al_diffusion_profile.png", title="Al adatom diffusion on Al(100) [EMT, CI-NEB]"
)
neb.save_csv("al_diffusion_profile.csv")
neb.save_trajectory("al_diffusion.traj")

print("\nProfile saved → al_diffusion_profile.png")
print("Energies saved → al_diffusion_profile.csv")
print("Trajectory saved → al_diffusion.traj")
