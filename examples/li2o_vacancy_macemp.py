"""
Li vacancy migration in bulk Li₂O — MACE-MP-0 calculator.

Crystal structure : Antifluorite (Fm-3m)
Mechanism        : Li⁺ vacancy hop between adjacent tetrahedral (8c) sites
                   via octahedral interstitial (4b) saddle point
Supercell        : 2×2×2 conventional cell, 95 atoms (96 - 1 vacancy)
Calculator       : MACE-MP-0 small, float64
Reference barrier: ~0.28 eV (DFT-GGA; range 0.20–0.35 eV)
nebwalk result   : 0.284 eV  (1.4% error)

Usage
-----
    python examples/li2o_vacancy_macemp.py

Runtime: ~10 minutes on CPU (4-core laptop).
"""

from copy import deepcopy

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.optimize import FIRE
from mace.calculators import mace_mp

from nebwalk import NEBRunConfig, run_neb_calculation

# ── Reference ─────────────────────────────────────────────────────────────────
REF_BARRIER = 0.28  # eV, DFT-GGA central estimate
REF_RANGE = (0.20, 0.35)

# ── Step 1: Build and relax Li₂O primitive cell ───────────────────────────────

a = 4.619  # Å, experimental lattice parameter

prim = Atoms(
    symbols=["O", "O", "O", "O", "Li", "Li", "Li", "Li", "Li", "Li", "Li", "Li"],
    scaled_positions=[
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
        [0.75, 0.75, 0.75],
        [0.25, 0.25, 0.75],
        [0.25, 0.75, 0.25],
        [0.75, 0.25, 0.25],
    ],
    cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
    pbc=True,
)

print("Step 1: Relaxing Li₂O primitive cell...")
calc = mace_mp(model="small", dispersion=False, default_dtype="float64")
prim.calc = calc
FIRE(prim, logfile=None).run(fmax=0.01)
a_rel = prim.cell.lengths()[0]
print(f"  Relaxed a = {a_rel:.4f} Å  (target: 4.55–4.65)")

# ── Step 2: Build 2×2×2 supercell and vacancy endpoints ──────────────────────

print("\nStep 2: Building supercell and vacancy endpoints...")
sc = make_supercell(prim, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
print(f"  Supercell: {len(sc)} atoms")

syms = sc.get_chemical_symbols()
pos = sc.get_positions()
li_idx = [i for i, s in enumerate(syms) if s == "Li"]

# Vacancy: Li nearest to supercell centre
centre = sc.cell[:].sum(axis=0) / 2
li_pos = np.array([pos[i] for i in li_idx])
vac_local = np.argmin(np.linalg.norm(li_pos - centre, axis=1))
vac_idx = li_idx[vac_local]

# Nearest-neighbour Li (hop destination)
nn = sorted(
    [(sc.get_distance(vac_idx, j, mic=True), j) for j in li_idx if j != vac_idx]
)
hop_idx = nn[0][1]
print(f"  Hop distance: {nn[0][0]:.4f} Å  (target: ~{a_rel / 2:.4f})")

# Save vacancy position before deletion
vac_pos = sc.positions[vac_idx].copy()

# Index of hopping Li in the vacancy structure (shifts if vac_idx < hop_idx)
hop_in_init = hop_idx - 1 if vac_idx < hop_idx else hop_idx

calc_i = mace_mp(model="small", dispersion=False, default_dtype="float64")
calc_f = mace_mp(model="small", dispersion=False, default_dtype="float64")

# Initial: remove vacancy site
print("  Relaxing initial state (vacancy at site A)...")
init = deepcopy(sc)
del init[vac_idx]
init.calc = calc_i
FIRE(init, logfile=None).run(fmax=0.02)
e_init = init.get_potential_energy()
print(f"    E = {e_init:.6f} eV   atoms = {len(init)}")

# Final: same atom ordering, move hopping Li to vacancy position
print("  Relaxing final state (vacancy at site B)...")
final = deepcopy(init)
final.positions[hop_in_init] = vac_pos
final.calc = calc_f
FIRE(final, logfile=None).run(fmax=0.02)
e_final = final.get_potential_energy()
print(f"    E = {e_final:.6f} eV   atoms = {len(final)}")

dE = abs(e_final - e_init) * 1000
print(f"\n  Endpoint ΔE = {dE:.2f} meV  (target: < 5 meV — symmetry check)")
if dE > 10:
    raise RuntimeError(
        f"Endpoint ΔE = {dE:.1f} meV > 10 meV. "
        "Structure preparation error — check atom ordering fix."
    )

# ── Step 3: CI-NEB ────────────────────────────────────────────────────────────

print("\nStep 3: Running CI-NEB (7 images, IDPP, MACE-MP-0)...")

config = NEBRunConfig(
    n_images=7,
    interpolation="idpp",
    k=0.5,
    climb=True,
    climb_delay=100,
    n_workers=1,
    fmax=0.05,
    max_steps=500,
    verbose=True,
)

result = run_neb_calculation(
    initial=init,
    final=final,
    calculator_factory=lambda: mace_mp(
        model="small", dispersion=False, default_dtype="float64"
    ),
    config=config,
)

# ── Results ───────────────────────────────────────────────────────────────────

energies = [img.get_potential_energy() for img in result.neb.images]
e0 = energies[0]

print("\n=== RESULTS ===")
print(f"Converged      : {result.converged}")
print(f"Forward barrier: {result.barrier * 1000:.1f} meV  ({result.barrier:.4f} eV)")
print(f"Reverse barrier: {result.reverse_barrier * 1000:.1f} meV")
print(f"Reaction energy: {result.reaction_energy * 1000:.2f} meV  (target: ~0)")
print(
    f"Reference DFT  : {REF_BARRIER * 1000:.0f} meV  "
    f"(range: {REF_RANGE[0] * 1000:.0f}–{REF_RANGE[1] * 1000:.0f} meV)"
)
print(f"Error          : {abs(result.barrier - REF_BARRIER) / REF_BARRIER * 100:.1f}%")

print("\nImage energies (relative to initial):")
for i, e in enumerate(energies):
    marker = " <- TS" if e == max(energies) else ""
    print(f"  [{i}] {(e - e0) * 1000:+8.1f} meV{marker}")

result.neb.plot("li2o_vacancy_profile.png")
result.neb.save_csv("li2o_vacancy_profile.csv")
result.neb.save_trajectory("li2o_vacancy_path.traj")
print(
    "\nSaved: li2o_vacancy_profile.png, li2o_vacancy_profile.csv, "
    "li2o_vacancy_path.traj"
)

# ── Attribution note ──────────────────────────────────────────────────────────
print("\nNote: MACE-MP-0 is well-trained on simple ionic oxides in Materials")
print("Project. Low error reflects in-distribution system, not algorithm")
print("precision — DFT-GGA reference itself has ±50 meV uncertainty.")
