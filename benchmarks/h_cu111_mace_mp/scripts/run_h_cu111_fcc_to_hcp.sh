#!/usr/bin/env bash
set -euo pipefail

# H diffusion on Cu(111) with NEBwalk + MACE-MP-0.
# This script runs:
#   Step 1: build and relax Cu(111)+H at fcc hollow
#   Step 2: create fcc and neighboring hcp endpoints
#   Step 3: run CI-NEB for the elementary fcc->hcp hop

python3 << 'EOF'
import json
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.io import write
from mace.calculators import mace_mp

A_CU = 3.6147
SIZE = (4, 4, 4)
VACUUM = 15.0
H_HEIGHT = 0.95
N_FREEZE_LAYERS = 2
FMAX = 0.015
MAX_STEPS = 1000
MACE_MODEL = "small"

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = get_device()

def make_calc():
    return mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64", device=DEVICE)

def distinct_layers(z_values, tol=0.35):
    groups = []
    for z in sorted(float(x) for x in z_values):
        if not groups or abs(z - np.mean(groups[-1])) > tol:
            groups.append([z])
        else:
            groups[-1].append(z)
    return [float(np.mean(g)) for g in groups]

def apply_bottom_constraint(atoms, n_freeze=N_FREEZE_LAYERS):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cu_z = [pos[2] for sym, pos in zip(symbols, positions) if sym == "Cu"]
    layers = distinct_layers(cu_z)
    cutoff = 0.5 * (layers[n_freeze - 1] + layers[n_freeze])
    mask = [(sym == "Cu" and pos[2] < cutoff) for sym, pos in zip(symbols, positions)]
    atoms.set_constraint(FixAtoms(mask=mask))
    return int(sum(mask))

print("=== Step 1: build and relax Cu(111)+H fcc ===")
slab = fcc111("Cu", size=SIZE, a=A_CU, vacuum=VACUUM, orthogonal=False)
add_adsorbate(slab, "H", height=H_HEIGHT, position="fcc", offset=(SIZE[0] // 2, SIZE[1] // 2))
slab.pbc = (True, True, False)
n_fixed = apply_bottom_constraint(slab)

print(f"Device       : {DEVICE}")
print(f"MACE model   : {MACE_MODEL}")
print(f"Fixed atoms  : {n_fixed}")
slab.calc = make_calc()
opt = FIRE(slab, logfile="01_Cu111_H_fcc_relax.log", trajectory="01_Cu111_H_fcc_relax.traj")
opt.run(fmax=FMAX, steps=MAX_STEPS)

energy = slab.get_potential_energy()
max_force = float(np.max(np.linalg.norm(slab.get_forces(), axis=1)))
write("Cu111_H_fcc_relaxed.vasp", slab, direct=True)
write("Cu111_H_fcc_relaxed.xyz", slab)

with open("01_Cu111_H_fcc_relaxed_metadata.json", "w") as f:
    json.dump({"energy_eV": float(energy), "max_force_eV_A": max_force}, f, indent=2)

print(f"Energy       : {energy:.8f} eV")
print(f"Max force    : {max_force:.5f} eV/Å")
EOF

python3 << 'EOF'
import json
import numpy as np
from ase import Atom
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.geometry import find_mic
from mace.calculators import mace_mp

N_FREEZE_LAYERS = 2
FMAX = 0.015
MAX_STEPS = 1000
MACE_MODEL = "small"

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = get_device()

def make_calc():
    return mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64", device=DEVICE)

def distinct_layers(z_values, tol=0.35):
    groups = []
    for z in sorted(float(x) for x in z_values):
        if not groups or abs(z - np.mean(groups[-1])) > tol:
            groups.append([z])
        else:
            groups[-1].append(z)
    return [float(np.mean(g)) for g in groups]

def apply_bottom_constraint(atoms, n_freeze=N_FREEZE_LAYERS):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cu_z = [pos[2] for sym, pos in zip(symbols, positions) if sym == "Cu"]
    layers = distinct_layers(cu_z)
    cutoff = 0.5 * (layers[n_freeze - 1] + layers[n_freeze])
    mask = [(sym == "Cu" and pos[2] < cutoff) for sym, pos in zip(symbols, positions)]
    atoms.set_constraint(FixAtoms(mask=mask))
    return int(sum(mask))

def get_h_index(atoms):
    h = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    if len(h) != 1:
        raise RuntimeError(f"Expected one H atom, found {len(h)}")
    return h[0]

def get_top_cu_indices(atoms, tol=0.35):
    cu_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Cu"]
    top_z = max(atoms.positions[i, 2] for i in cu_idx)
    return [i for i in cu_idx if abs(atoms.positions[i, 2] - top_z) < tol]

def get_neighboring_hcp_position_from_fcc(atoms, h_idx, edge_choice=0):
    h_pos = atoms.positions[h_idx].copy()
    top_idx = get_top_cu_indices(atoms)
    local = []
    for i in top_idx:
        vec, dist = find_mic(atoms.positions[i] - h_pos, atoms.cell, atoms.pbc)
        local.append((dist, i, h_pos + vec))
    local.sort(key=lambda x: x[0])
    tri_pos = [x[2] for x in local[:3]]
    edges = [(0, 1), (1, 2), (2, 0)]
    a, b = edges[edge_choice % 3]
    edge_mid = 0.5 * (tri_pos[a] + tri_pos[b])
    hcp_pos = h_pos + 2.0 * (edge_mid - h_pos)
    hcp_pos[2] = h_pos[2]
    return hcp_pos

print("=== Step 2: create fcc and hcp endpoints ===")
relaxed = read("Cu111_H_fcc_relaxed.vasp")
relaxed.pbc = (True, True, False)
h_old = get_h_index(relaxed)
h_pos = relaxed.positions[h_old].copy()

clean = relaxed[[i for i, s in enumerate(relaxed.get_chemical_symbols()) if s != "H"]]
clean.pbc = (True, True, False)
apply_bottom_constraint(clean)

initial = clean.copy()
initial.append(Atom("H", h_pos))
initial.pbc = (True, True, False)
apply_bottom_constraint(initial)
initial.wrap()
initial.calc = make_calc()
FIRE(initial, logfile="02_Cu111_fcc_initial_relax.log", trajectory="02_Cu111_fcc_initial_relax.traj").run(fmax=FMAX, steps=MAX_STEPS)
e_initial = initial.get_potential_energy()
write("Cu111_fcc_initial.vasp", initial, direct=True)

h_i = get_h_index(initial)
hcp_pos = get_neighboring_hcp_position_from_fcc(initial, h_i)

final = clean.copy()
final.append(Atom("H", hcp_pos))
final.pbc = (True, True, False)
apply_bottom_constraint(final)
final.wrap()
final.calc = make_calc()
FIRE(final, logfile="02_Cu111_hcp_final_relax.log", trajectory="02_Cu111_hcp_final_relax.traj").run(fmax=FMAX, steps=MAX_STEPS)
e_final = final.get_potential_energy()
write("Cu111_hcp_final.vasp", final, direct=True)

h_f = get_h_index(final)
_, hop_dist = find_mic(final.positions[h_f] - initial.positions[h_i], initial.cell, initial.pbc)
reaction_meV = (e_final - e_initial) * 1000.0

with open("02_Cu111_fcc_to_hcp_endpoints_metadata.json", "w") as f:
    json.dump({
        "initial_energy_eV": float(e_initial),
        "final_energy_eV": float(e_final),
        "reaction_energy_meV": float(reaction_meV),
        "relaxed_hop_distance_A": float(hop_dist),
    }, f, indent=2)

print(f"Initial fcc E   : {e_initial:.8f} eV")
print(f"Final hcp E     : {e_final:.8f} eV")
print(f"Reaction energy : {reaction_meV:+.3f} meV")
print(f"H hop distance  : {hop_dist:.3f} Å")
EOF

python3 << 'EOF'
import json
import time
import numpy as np
from ase.io import read
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
from NEBwalk import NEBRunConfig, run_neb_calculation

N_FREEZE_LAYERS = 2
N_IMAGES = 5
SPRING_K = 0.5
FMAX_NEB = 0.03
MAX_STEPS = 1000
CLIMB_DELAY = 50
MACE_MODEL = "small"

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DEVICE = get_device()

def make_calc():
    return mace_mp(model=MACE_MODEL, dispersion=False, default_dtype="float64", device=DEVICE)

def distinct_layers(z_values, tol=0.35):
    groups = []
    for z in sorted(float(x) for x in z_values):
        if not groups or abs(z - np.mean(groups[-1])) > tol:
            groups.append([z])
        else:
            groups[-1].append(z)
    return [float(np.mean(g)) for g in groups]

def apply_bottom_constraint(atoms, n_freeze=N_FREEZE_LAYERS):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cu_z = [pos[2] for sym, pos in zip(symbols, positions) if sym == "Cu"]
    layers = distinct_layers(cu_z)
    cutoff = 0.5 * (layers[n_freeze - 1] + layers[n_freeze])
    mask = [(sym == "Cu" and pos[2] < cutoff) for sym, pos in zip(symbols, positions)]
    atoms.set_constraint(FixAtoms(mask=mask))

print("=== Step 3: CI-NEB fcc -> hcp ===")
initial = read("Cu111_fcc_initial.vasp")
final = read("Cu111_hcp_final.vasp")
initial.pbc = (True, True, False)
final.pbc = (True, True, False)
apply_bottom_constraint(initial)
apply_bottom_constraint(final)

config = NEBRunConfig(
    n_images=N_IMAGES,
    interpolation="idpp",
    k=SPRING_K,
    climb=True,
    climb_delay=CLIMB_DELAY,
    n_workers=1,
    fmax=FMAX_NEB,
    max_steps=MAX_STEPS,
    verbose=True,
)

t0 = time.time()
result = run_neb_calculation(initial, final, make_calc, config)
elapsed_min = (time.time() - t0) / 60.0

energies = []
for img in result.neb.images:
    if img.calc is None:
        img.calc = make_calc()
    energies.append(float(img.get_potential_energy()))
e0 = energies[0]
rel_meV = [(e - e0) * 1000.0 for e in energies]
ts_index = int(np.argmax(energies))

result.neb.plot("Cu111_H_fcc_to_hcp_profile.png")
result.neb.save_csv("Cu111_H_fcc_to_hcp_profile.csv")
result.neb.save_trajectory("Cu111_H_fcc_to_hcp_path.traj")

metadata = {
    "converged": bool(result.converged),
    "wall_time_min": float(elapsed_min),
    "forward_barrier_eV": float(result.barrier),
    "reverse_barrier_eV": float(result.reverse_barrier),
    "reaction_energy_eV": float(result.reaction_energy),
    "ts_image_index": ts_index,
    "relative_energies_meV": [float(x) for x in rel_meV],
}

with open("Cu111_H_fcc_to_hcp_CI_NEB_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Converged       : {result.converged}")
print(f"Forward barrier : {result.barrier * 1000:.2f} meV")
print(f"Reverse barrier : {result.reverse_barrier * 1000:.2f} meV")
print(f"Reaction energy : {result.reaction_energy * 1000:+.3f} meV")
for i, e in enumerate(rel_meV):
    marker = "  <-- TS" if i == ts_index else ""
    print(f"image {i:02d}: {e:+10.3f} meV{marker}")
EOF
