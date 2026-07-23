"""
Cu adatom diffusion on Cu(100) — EMT calculator.

System     : Cu adatom hopping between adjacent 4-fold hollow sites
             on a Cu(100) surface slab (3×3×4 layers, 10 Å vacuum)
Calculator : EMT (Effective Medium Theory)
Reference  : ~0.40 eV (DFT-LDA/experiment; Feibelman, PRB 1999)
nebwalk    : 0.418 eV  (4.6% error)

Note: EMT ignores d-band and relativistic effects. Error reflects
calculator limitation, not algorithm error.

Usage
-----
    python examples/cu_adatom_cu100_emt.py

Runtime: < 1 minute on any CPU.
"""

import numpy as np
from ase import Atom
from ase.build import fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from nebwalk import NEBRunConfig, run_neb_calculation

# ── Reference ─────────────────────────────────────────────────────────────────
REF_BARRIER = 0.40  # eV, DFT-LDA/FIM experiment

# ── Step 1: Build and relax clean Cu(100) slab ────────────────────────────────

print("Step 1: Building Cu(100) slab (3×3×4 layers, 10 Å vacuum)...")
slab = fcc100("Cu", size=(3, 3, 4), vacuum=10.0)

# Fix bottom 2 layers
z_sorted = np.sort(np.unique(np.round(slab.positions[:, 2], 2)))
z_fix = z_sorted[1]
mask = slab.positions[:, 2] <= z_fix + 0.1
slab.set_constraint(FixAtoms(mask=mask))
print(f"  {len(slab)} slab atoms, {mask.sum()} fixed (bottom 2 layers)")

slab.calc = EMT()
FIRE(slab, logfile=None).run(fmax=0.02)
print("  Clean slab relaxed.")

# Surface geometry
z_top = slab.positions[:, 2].max()
x_s = slab.cell[0, 0] / 3  # surface unit cell spacing ≈ a/√2 = 2.556 Å
y_s = slab.cell[1, 1] / 3
print(f"  Surface unit cell: {x_s:.4f} × {y_s:.4f} Å  (expected: ~2.556)")

# Adjacent 4-fold hollow sites
pos_A = np.array([0.5 * x_s, 0.5 * y_s, z_top + 1.8])
pos_B = np.array([1.5 * x_s, 0.5 * y_s, z_top + 1.8])
hop_dist = np.linalg.norm(pos_B - pos_A)
print(f"  Hollow A → Hollow B distance: {hop_dist:.4f} Å  (= surface unit cell)")

# ── Step 2: Build and relax endpoints ─────────────────────────────────────────

print("\nStep 2: Relaxing endpoints...")


def make_endpoint(base_slab, adatom_pos):
    """Append adatom and relax. Base slab constraints are preserved."""
    s = base_slab.copy()
    s.append(Atom("Cu", position=adatom_pos))
    s.calc = EMT()
    return s


init = make_endpoint(slab, pos_A)
FIRE(init, logfile=None).run(fmax=0.02)
e_init = init.get_potential_energy()
print(f"  Initial: E = {e_init:.6f} eV   adatom z = {init.positions[-1, 2]:.3f} Å")

final = make_endpoint(slab, pos_B)
FIRE(final, logfile=None).run(fmax=0.02)
e_final = final.get_potential_energy()
print(f"  Final  : E = {e_final:.6f} eV   adatom z = {final.positions[-1, 2]:.3f} Å")

dE = abs(e_final - e_init) * 1000
print(f"\n  Endpoint ΔE = {dE:.2f} meV  (target: < 5 meV — equivalent hollow sites)")
if dE > 10:
    raise RuntimeError(f"ΔE = {dE:.1f} meV > 10 meV. Check hollow site geometry.")

# ── Step 3: CI-NEB ────────────────────────────────────────────────────────────

print("\nStep 3: Running CI-NEB (7 images, IDPP, EMT)...")

config = NEBRunConfig(
    n_images=7,
    interpolation="idpp",
    k=0.5,
    climb=True,
    climb_delay=50,
    n_workers=1,
    fmax=0.05,
    max_steps=300,
    verbose=True,
)

result = run_neb_calculation(
    initial=init,
    final=final,
    calculator_factory=lambda: EMT(),
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
print(f"Reference      : {REF_BARRIER * 1000:.0f} meV  (DFT-LDA/FIM experiment)")
print(f"Error          : {abs(result.barrier - REF_BARRIER) / REF_BARRIER * 100:.1f}%")

print("\nImage energies (relative to initial):")
for i, e in enumerate(energies):
    marker = " <- TS" if e == max(energies) else ""
    print(f"  [{i}] {(e - e0) * 1000:+7.1f} meV{marker}")

result.neb.plot("cu_adatom_cu100_profile.png")
result.neb.save_csv("cu_adatom_cu100_profile.csv")
result.neb.save_trajectory("cu_adatom_cu100_path.traj")
print(
    "\nSaved: cu_adatom_cu100_profile.png, cu_adatom_cu100_profile.csv, "
    "cu_adatom_cu100_path.traj"
)

print("\nNote: DFT-LDA reference ~0.40 eV has ±0.05 eV spread in literature.")
print("EMT captures hollow-site energy difference but not d-band detail at TS.")
