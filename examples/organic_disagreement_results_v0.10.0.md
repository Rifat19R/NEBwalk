# Organic Cross-Model Disagreement Runs (v0.10.0)

All runs used Egret-1t as the primary calculator and MACE-OFF23 medium as the
secondary calculator with `default_dtype="float64"`. Selection strategy was
`uncertainty_disagreement` with the default `force_disagreement` metric.

These are exploratory organic-domain runs, not calibrated uncertainty
quantification and not DFT validation. Acetaldehyde was tested but excluded
from the pushed example set because its barrier was a clear outlier and its
default force-disagreement selection was endpoint-adjacent.

## Summary

| System | Script | Barrier (eV) | Selected images |
| --- | --- | ---: | --- |
| Ethane torsion | `ethane_egret_maceoff23_disagreement.py` | 0.109619 | (1, 4, 7) |
| Propane terminal methyl torsion | `propane_egret_maceoff23_disagreement.py` | 0.124756 | (3, 4, 5) |
| Methanol hydroxyl torsion | `methanol_egret_maceoff23_disagreement.py` | 0.053955 | (1, 3, 7) |
| Ethanol C-C torsion | `ethanol_egret_maceoff23_disagreement.py` | 0.121582 | (1, 4, 7) |
| Dimethyl ether methyl torsion | `dimethyl_ether_egret_maceoff23_disagreement.py` | 0.101074 | (3, 4, 5) |

## Per-Image Diagnostics

### Ethane torsion

| Image | Valid | dE_rel (eV) | dF_max (eV/A) |
| ---: | :---: | ---: | ---: |
| 00 | True | +0.00000000 | 0.00946898 |
| 01 | True | +0.00018635 | 0.00855045 |
| 02 | True | +0.00043259 | 0.00831366 |
| 03 | True | +0.00168986 | 0.00794919 |
| 04 | True | +0.00443009 | 0.00874598 |
| 05 | True | +0.00168986 | 0.00795144 |
| 06 | True | +0.00043258 | 0.00830975 |
| 07 | True | +0.00018636 | 0.00854888 |
| 08 | True | +0.00000000 | 0.00946901 |

### Propane terminal methyl torsion

| Image | Valid | dE_rel (eV) | dF_max (eV/A) |
| ---: | :---: | ---: | ---: |
| 00 | True | +0.00000000 | 0.00655213 |
| 01 | True | +0.00039899 | 0.00702104 |
| 02 | True | +0.00082414 | 0.00761024 |
| 03 | True | +0.00248675 | 0.01023524 |
| 04 | True | +0.00451041 | 0.01373919 |
| 05 | True | +0.00252685 | 0.01030879 |
| 06 | True | +0.00078220 | 0.00757850 |
| 07 | True | +0.00026223 | 0.00697263 |
| 08 | True | +0.00004211 | 0.00655688 |

### Methanol hydroxyl torsion

| Image | Valid | dE_rel (eV) | dF_max (eV/A) |
| ---: | :---: | ---: | ---: |
| 00 | True | +0.00000000 | 0.01279737 |
| 01 | True | +0.00030447 | 0.01411896 |
| 02 | True | +0.00025697 | 0.01370921 |
| 03 | True | +0.00026321 | 0.01469294 |
| 04 | True | +0.00017273 | 0.01363493 |
| 05 | True | +0.00037437 | 0.01372560 |
| 06 | True | +0.00025230 | 0.01400237 |
| 07 | True | +0.00030513 | 0.01434321 |
| 08 | True | +0.00112763 | 0.01352124 |

### Ethanol C-C torsion

| Image | Valid | dE_rel (eV) | dF_max (eV/A) |
| ---: | :---: | ---: | ---: |
| 00 | True | +0.00000000 | 0.01290567 |
| 01 | True | +0.00047003 | 0.01920781 |
| 02 | True | +0.00055043 | 0.01912220 |
| 03 | True | +0.00074920 | 0.01566946 |
| 04 | True | +0.00046695 | 0.02537435 |
| 05 | True | +0.00067898 | 0.01564300 |
| 06 | True | +0.00011472 | 0.01911988 |
| 07 | True | +0.00049131 | 0.01916896 |
| 08 | True | +0.00000950 | 0.01329790 |

### Dimethyl ether methyl torsion

| Image | Valid | dE_rel (eV) | dF_max (eV/A) |
| ---: | :---: | ---: | ---: |
| 00 | True | +0.00000000 | 0.00649454 |
| 01 | True | +0.00001018 | 0.00758300 |
| 02 | True | +0.00027807 | 0.01241170 |
| 03 | True | +0.00217846 | 0.01892268 |
| 04 | True | +0.00341284 | 0.02405957 |
| 05 | True | +0.00179277 | 0.01837497 |
| 06 | True | +0.00027868 | 0.01169521 |
| 07 | True | +0.00044762 | 0.00727635 |
| 08 | True | +0.00031260 | 0.00783051 |
