"""
Example 1: Collinear H-atom exchange  HA•••HB-HC → HA-HB•••HC

Uses a simplified SEAM (Sum of Morse pairs) potential for demonstration.
This is NOT physically accurate for H+H2; the LEPS potential would be needed
for quantitative barriers.  The purpose here is to verify that NEBwalk finds
a smooth saddle-point profile without any external calculator.

Key point: HA (index 0) and HC (index 2) are fixed across all intermediate
images via FixAtoms — only HB (index 1) moves along the reaction coordinate.
Without these constraints the NEB optimizer would also move the terminal atoms,
finding a spurious barrierless path.

Run:
    python examples/morse_h3.py
"""

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from NEBwalk import NEB, linear_interpolate

# ---------------------------------------------------------------------------
# Morse potential parameters for H2
# D   = 4.52  eV    (De, dissociation energy)
# a   = 1.93  Å⁻¹
# r0  = 0.74  Å     (equilibrium H-H bond length)
# ---------------------------------------------------------------------------

D, A, R0 = 4.52, 1.93, 0.74


def _morse_pair(r):
    """Energy (eV) and scalar dV/dr (eV/Å) for a single Morse pair."""
    u = np.exp(-A * (r - R0))
    E = D * (1.0 - u) ** 2 - D
    dEdr = 2.0 * D * A * u * (1.0 - u)
    return E, dEdr


class MorsePairH3:
    """
    SEAM calculator for the collinear H3 system:
        E = V_Morse(r_HA-HB) + V_Morse(r_HB-HC)

    Atom ordering: 0=HA, 1=HB, 2=HC — all on the x-axis.
    HA-HC interaction is neglected (they are far apart at the endpoints).
    """

    def get_potential_energy(self, atoms, force_consistent=False):
        r_AB = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
        r_BC = float(np.linalg.norm(atoms.positions[2] - atoms.positions[1]))
        E_AB, _ = _morse_pair(r_AB)
        E_BC, _ = _morse_pair(r_BC)
        return E_AB + E_BC

    def get_forces(self, atoms):
        pos = atoms.positions
        r_vec_AB = pos[1] - pos[0]
        r_vec_BC = pos[2] - pos[1]
        r_AB = float(np.linalg.norm(r_vec_AB))
        r_BC = float(np.linalg.norm(r_vec_BC))
        _, dEdr_AB = _morse_pair(r_AB)
        _, dEdr_BC = _morse_pair(r_BC)

        r_hat_AB = r_vec_AB / r_AB  # unit vector HA→HB
        r_hat_BC = r_vec_BC / r_BC  # unit vector HB→HC

        f = np.zeros((3, 3))
        # F_A = -dV/dr_A = -dV_AB/dr * (dr/dr_A) = +dEdr * r_hat_AB
        # (positive dEdr at r > r0 → force on HA toward HB = attractive)
        f[0] += dEdr_AB * r_hat_AB
        f[1] -= dEdr_AB * r_hat_AB  # reaction on HB from AB
        f[1] += dEdr_BC * r_hat_BC  # HB pulled toward HC
        f[2] -= dEdr_BC * r_hat_BC  # reaction on HC from BC
        return f


# ---------------------------------------------------------------------------
# Build start and end states
# ---------------------------------------------------------------------------

# Start: HA-HB bonded at r0 = 0.74 Å; HC separated at 2.50 Å from HA
start = Atoms("H3", positions=[[0.00, 0, 0], [0.74, 0, 0], [2.50, 0, 0]])
start.calc = MorsePairH3()

# End: HB-HC bonded; HA now separated
end = Atoms(
    "H3",
    positions=[
        [0.00, 0, 0],
        [1.76, 0, 0],  # 2.50 - 0.74
        [2.50, 0, 0],
    ],
)
end.calc = MorsePairH3()

print(f"Start energy : {start.get_potential_energy():.4f} eV")
print(f"End   energy : {end.get_potential_energy():.4f} eV")
print("Analytical TS barrier: ~0.19 eV  (at r_AB = r_BC = 1.25 Å)")

# ---------------------------------------------------------------------------
# NEB — with FixAtoms on HA and HC in every intermediate image
# ---------------------------------------------------------------------------

N_IMAGES = 7
images = linear_interpolate(start, end, n_images=N_IMAGES)

for img in images:
    img.calc = MorsePairH3()

# Fix HA (index 0) and HC (index 2) in intermediate images so only HB moves
for img in images[1:-1]:
    img.set_constraint(FixAtoms(indices=[0, 2]))

neb = NEB(images, k=0.50, climb=True, climb_delay=60)
converged = neb.optimize(fmax=0.05, max_steps=500)

print(f"\nConverged : {converged}")
print(f"Forward barrier : {neb.get_barrier():.4f} eV  (expected ~0.19 eV)")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

neb.plot("morse_h3_profile.png", title="H+H₂ collinear exchange (SEAM Morse)")
neb.save_csv("morse_h3_profile.csv")
neb.save_trajectory("morse_h3.traj")
