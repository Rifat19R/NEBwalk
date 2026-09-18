# H/Cu(111) Surface Diffusion Benchmark

**Calculator:** MACE-MP-0 (small, float64)
**NEBwalk version:** 0.6.0
**System:** H adatom diffusion on Cu(111) surface

## Mechanisms studied

### fcc → hcp (asymmetric hop)
H moves from FCC hollow to adjacent HCP hollow site.

| Quantity | Value |
|---|---|
| Forward barrier | 125.92 meV |
| Reverse barrier | 125.56 meV |
| Reaction energy | 0.36 meV |
| Converged | True |

### fcc → fcc (symmetric hop via bridge)
H moves between equivalent FCC hollow sites through bridge site.

| Quantity | Value |
|---|---|
| Forward barrier | 125.56 meV |
| Reverse barrier | 125.56 meV |
| Reaction energy | 0.00 meV |
| Converged | True |

## Reference
DFT-PBE literature value: ~130–160 meV (Hammer & Norskov 1995;
Michaelides et al. 2003). MACE-MP-0 error: ~15–20% — expected for a
universal potential on surface adsorbate chemistry.

## Reproduce

```bash
pip install NEBwalk
cd scripts/
bash run_h_cu111_fcc_to_hcp.sh
```

## Attribution
Calculator failure note: MACE-MP-0 is trained primarily on bulk Materials
Project structures. Surface + adsorbate systems are less represented in
training data. Results here are documented as exploratory benchmarks,
not production-quality barriers.
