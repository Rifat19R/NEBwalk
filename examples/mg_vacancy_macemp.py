"""
Mg vacancy migration (basal plane) in HCP Mg — MACE-MP-0 calculator.

The vacancy hops between nearest-neighbour sites within the basal (0001)
plane. NN distance = a = 3.209 Å.

Reference barrier (basal hop, DFT-PBE): ~0.52 eV.
See vacancy_benchmark_suite.SYSTEMS["mg"] for the same reference value.

Usage
-----
    python examples/mg_vacancy_macemp.py
"""

import numpy as np
from ase.build import bulk
from ase.optimize import BFGS
from mace.calculators import mace_mp

from nebwalk import NEB, linear_interpolate


def make_calc():
    return mace_mp(
        model="small", dispersion=False, default_dtype="float32", device="cpu"
    )


# ---------------------------------------------------------------------------
# 1. Build HCP Mg supercell and create vacancy
# ---------------------------------------------------------------------------

mg_bulk = bulk("Mg", "hcp", a=3.209, c=5.211)
supercell = mg_bulk.repeat([3, 3, 2])  # 3x3x2 → 36 atoms

vacancy_index = 0
vacancy_pos = supercell.positions[vacancy_index].copy()
atom0_z = vacancy_pos[2]  # z-coordinate of vacancy layer
del supercell[vacancy_index]  # 35 atoms remain

# ---------------------------------------------------------------------------
# 2. Find nearest basal-plane neighbour
#    Filter to same z-layer (|Δz| < 0.5 Å), then find minimum distance.
#    Basal NN distance = a = 3.209 Å.
# ---------------------------------------------------------------------------

same_layer = np.abs(supercell.positions[:, 2] - atom0_z) < 0.5
distances = np.linalg.norm(supercell.positions - vacancy_pos, axis=1)
distances_basal = np.where(same_layer, distances, np.inf)
nn_index = int(np.argmin(distances_basal))
nn_dist = distances[nn_index]
print(f"Migrating atom : index {nn_index}")
print(f"NN distance    : {nn_dist:.4f} Å  (expected ~3.209 Å for HCP Mg basal plane)")

# ---------------------------------------------------------------------------
# 3. Build and relax endpoints
# ---------------------------------------------------------------------------

initial = supercell.copy()
initial.calc = make_calc()

final = supercell.copy()
final.positions[nn_index] = vacancy_pos
final.calc = make_calc()

BFGS(initial, logfile=None).run(fmax=0.02)
BFGS(final, logfile=None).run(fmax=0.02)

print(f"\nInitial energy : {initial.get_potential_energy():.4f} eV")
print(f"Final energy   : {final.get_potential_energy():.4f} eV")
print(
    f"ΔE             : "
    f"{final.get_potential_energy() - initial.get_potential_energy():.5f} eV"
)

# ---------------------------------------------------------------------------
# 4. Build NEB path
# ---------------------------------------------------------------------------

images = linear_interpolate(initial, final, n_images=7)
for img in images:
    img.calc = make_calc()

# ---------------------------------------------------------------------------
# 5. Run NEB
# ---------------------------------------------------------------------------

neb = NEB(images, k=0.1, climb=True, climb_delay=30)
converged = neb.optimize(fmax=0.05, max_steps=300)

# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------

barrier = neb.get_barrier()
ref = 0.52
print(f"\nConverged      : {converged}")
print(f"Forward barrier: {barrier:.4f} eV")
print(f"Reference (DFT): ~{ref} eV  (basal plane, DFT-PBE)")
print(f"Error          : {abs(barrier - ref) / ref * 100:.1f}%")

neb.plot(
    "mg_vacancy_macemp_profile.png",
    title="Mg vacancy migration in HCP Mg [MACE-MP-0, CI-NEB]",
)
neb.save_csv("mg_vacancy_macemp_profile.csv")
neb.save_trajectory("mg_vacancy_macemp.traj")
