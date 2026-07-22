"""
Mg vacancy migration in bulk MgO — MACE-MP-0 calculator.

Crystal structure : Rock salt (Fm-3m)
Mechanism        : Mg²⁺ vacancy hop between nearest-neighbour sites
                   via triangular gate of O²⁻ ions (saddle point)
Supercell        : 2×2×2 conventional cell, 63 atoms (64 - 1 vacancy)
Calculator       : MACE-MP-0 small, float64
Reference barrier: ~2.2 eV (DFT-PBE; range 1.9–2.5 eV)
nebwalk result   : 2.254 eV  (2.5% error)

Usage
-----
    python examples/mg_vacancy_mgo_macemp.py

Runtime: ~10 minutes on CPU.

Note: The steep energy rise near the saddle point is real physics.
The migrating Mg²⁺ must pass through a triangular gate of three O²⁻
ions — strongly ionic repulsion produces the sharp peak. Uniform
spring constants place fewer images near the TS; variable springs
(k_min set) would improve image distribution but do not change the
CI-NEB barrier value.
"""

from copy import deepcopy

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.optimize import FIRE
from mace.calculators import mace_mp

from nebwalk import NEBRunConfig, run_neb_calculation

# ── Reference ─────────────────────────────────────────────────────────────────
REF_BARRIER = 2.20   # eV, DFT-PBE central estimate
REF_RANGE   = (1.90, 2.50)

# ── Step 1: Build and relax MgO primitive cell ────────────────────────────────

print("Step 1: Building and relaxing MgO primitive cell...")

a = 4.211  # Å, experimental

prim = Atoms(
    symbols=["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
    scaled_positions=[
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5], [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0], [0.5, 0.0, 0.0],
    ],
    cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
    pbc=True,
)

calc = mace_mp(model="small", dispersion=False, default_dtype="float64")
prim.calc = calc
FIRE(prim, logfile=None).run(fmax=0.01)

a_rel  = prim.cell.lengths()[0]
mg_idx = [i for i, s in enumerate(prim.get_chemical_symbols()) if s == "Mg"]
o_idx  = [i for i, s in enumerate(prim.get_chemical_symbols()) if s == "O"]
mg_o   = sorted([prim.get_distance(i, j, mic=True)
                 for i in mg_idx for j in o_idx])

print(f"  Relaxed a  = {a_rel:.4f} Å  (target: 4.18–4.25)")
print(f"  Mg-O dist  = {mg_o[0]:.4f} Å  (target: ~2.11)")

# ── Step 2: Build 2×2×2 supercell and Mg vacancy endpoints ───────────────────

print("\nStep 2: Building supercell and Mg vacancy endpoints...")
sc = make_supercell(prim, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
print(f"  Supercell: {len(sc)} atoms  (32 Mg + 32 O)")

syms   = sc.get_chemical_symbols()
pos    = sc.get_positions()
mg_idx = [i for i, s in enumerate(syms) if s == "Mg"]

# Vacancy: Mg nearest to supercell centre
centre    = sc.cell[:].sum(axis=0) / 2
mg_pos    = np.array([pos[i] for i in mg_idx])
vac_local = np.argmin(np.linalg.norm(mg_pos - centre, axis=1))
vac_idx   = mg_idx[vac_local]

# Nearest-neighbour Mg (hop destination; = a/√2 ≈ 2.98 Å in rock salt)
nn      = sorted([(sc.get_distance(vac_idx, j, mic=True), j)
                  for j in mg_idx if j != vac_idx])
hop_idx = nn[0][1]
print(f"  Hop distance: {nn[0][0]:.4f} Å  (target: ~2.98 = a/√2)")

# Save vacancy position before deletion
vac_pos = sc.positions[vac_idx].copy()

# Index of hopping Mg in the vacancy structure
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

# Final: same atom ordering, move hopping Mg to vacancy position
print("  Relaxing final state (vacancy at site B)...")
final = deepcopy(init)
final.positions[hop_in_init] = vac_pos
final.calc = calc_f
FIRE(final, logfile=None).run(fmax=0.02)
e_final = final.get_potential_energy()
print(f"    E = {e_final:.6f} eV   atoms = {len(final)}")

dE = abs(e_final - e_init) * 1000
d_hop = np.linalg.norm(
    final.positions[hop_in_init] - init.positions[hop_in_init]
)
print(f"\n  Endpoint ΔE    = {dE:.2f} meV  (target: < 5 meV — symmetry check)")
print(f"  Mg displacement = {d_hop:.4f} Å   (target: ~2.98)")
if dE > 10:
    raise RuntimeError(
        f"Endpoint ΔE = {dE:.1f} meV > 10 meV. "
        "Structure preparation error — check atom ordering fix."
    )

# ── Step 3: CI-NEB ────────────────────────────────────────────────────────────

print("\nStep 3: Running CI-NEB (7 images, IDPP, MACE-MP-0)...")

config = NEBRunConfig(
    n_images      = 7,
    interpolation = "idpp",
    k             = 0.5,
    climb         = True,
    climb_delay   = 100,
    n_workers     = 1,
    fmax          = 0.05,
    max_steps     = 500,
    verbose       = True,
)

result = run_neb_calculation(
    initial            = init,
    final              = final,
    calculator_factory = lambda: mace_mp(
        model="small", dispersion=False, default_dtype="float64"
    ),
    config             = config,
)

# ── Results ───────────────────────────────────────────────────────────────────

energies = [img.get_potential_energy() for img in result.neb.images]
e0       = energies[0]

print("\n=== RESULTS ===")
print(f"Converged      : {result.converged}")
print(f"Forward barrier: {result.barrier*1000:.1f} meV  ({result.barrier:.4f} eV)")
print(f"Reverse barrier: {result.reverse_barrier*1000:.1f} meV")
print(f"Reaction energy: {result.reaction_energy*1000:.2f} meV  (target: ~0)")
print(f"Reference DFT  : {REF_BARRIER*1000:.0f} meV  "
      f"(range: {REF_RANGE[0]*1000:.0f}–{REF_RANGE[1]*1000:.0f} meV)")
print(f"Error          : {abs(result.barrier - REF_BARRIER)/REF_BARRIER*100:.1f}%")

print("\nImage energies (relative to initial):")
for i, e in enumerate(energies):
    marker = " <- TS" if e == max(energies) else ""
    print(f"  [{i}] {(e - e0)*1000:+8.1f} meV{marker}")

result.neb.plot("mg_vacancy_mgo_macemp_profile.png")
result.neb.save_csv("mg_vacancy_mgo_macemp_profile.csv")
result.neb.save_trajectory("mg_vacancy_mgo_macemp_path.traj")
print("\nSaved: mg_vacancy_mgo_macemp_profile.png, mg_vacancy_mgo_macemp_profile.csv, "
      "mg_vacancy_mgo_macemp_path.traj")

print("\nNote: Steep rise near TS is physical — Mg²⁺ passing through O²⁻ gate.")
print("MACE-MP-0 is well-trained on simple ionic oxides in Materials Project.")
