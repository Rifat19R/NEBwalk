# Organic Disagreement Runs vs Literature-Scale Barriers

nebwalk v0.10.0 examples used Egret-1t as the primary calculator and
MACE-OFF23 medium as the secondary calculator with `default_dtype="float64"`.
Selection strategy was `uncertainty_disagreement` with the default
`force_disagreement` metric.

These are exploratory organic-domain checks, not DFT validation and not
calibrated uncertainty quantification. Literature values below are
literature-scale gas-phase torsional/internal-rotor references. Some are
spectroscopic internal-rotor `V3` parameters rather than the exact same NEB
endpoint-to-endpoint path. Treat percent differences as scale diagnostics, not
formal benchmark errors.

Acetaldehyde was tested but intentionally excluded from the pushed example set:
its barrier was about 42% below the literature-scale methyl-rotation target,
and the default force-disagreement selector chose endpoint-adjacent images.
That makes it a useful future stress test, not defensible validation evidence.

Conversion used: `1 eV = 96.485 kJ/mol = 23.061 kcal/mol`.

## Summary

| System | Torsion path in script | nebwalk barrier (eV) | nebwalk barrier (kJ/mol) | Selected images | Literature-scale target | Difference vs target | Verdict |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| Ethane | C-C methyl torsion | 0.109619 | 10.58 | (1, 4, 7) | ~0.1296 eV / 12.5 kJ/mol | -15.4% | Good scale; slightly low |
| Propane | terminal methyl C-C torsion | 0.124756 | 12.04 | (3, 4, 5) | ~0.140 eV / ~13.5 kJ/mol | -10.9% | Good scale |
| Methanol | O-H rotation about C-O | 0.053955 | 5.21 | (1, 3, 7) | ~0.0463 eV / ~4.47 kJ/mol (`V3` ~373.5 cm^-1) | +16.5% | Plausible scale; flat signal |
| Ethanol | C-C torsion | 0.121582 | 11.73 | (1, 4, 7) | ~0.12 eV / ~11.6 kJ/mol rough conformer-scale target | +1.3% | Good scale, but path-specific |
| Dimethyl ether | methyl rotation about C-O | 0.101074 | 9.75 | (3, 4, 5) | ~0.11 eV / ~10.7 kJ/mol internal-rotor scale | -8.1% | Good scale |

## Per-System Notes

### Ethane

- Literature expectation: staggered ethane lower than eclipsed by about
  12.5 kJ/mol.
- nebwalk gives 10.58 kJ/mol, about 15% low.
- Energy disagreement peaks at image 4 (`+0.00443009 eV`), matching the
  expected eclipsed region.
- Default force-disagreement selection picks `(1, 4, 7)` because force
  disagreement at image 1/7 slightly exceeds image 3/5. This is not a crash or
  physics failure; it is a consequence of using `force_disagreement`.

### Propane

- Literature expectation: propane has staggered minima and eclipsed transition
  states around C-C rotation, similar to ethane but with slightly larger
  substituent effects.
- nebwalk gives 12.04 kJ/mol, close to the common ~13-14 kJ/mol scale.
- Both energy and force disagreement peak around image 4.
- Selected `(3, 4, 5)` is the cleanest pattern in this set.

### Methanol

- Literature expectation: methanol is a classic internal-rotor system; useful
  spectroscopic descriptions use torsional constants including `V3`.
- nebwalk gives 5.21 kJ/mol, close to the internal-rotor scale around
  4.5 kJ/mol.
- Signal is shallow. Selected `(1, 3, 7)` is force-driven, not a clean
  energy-peak cluster.
- Use as sanity check only, not validation claim.

### Ethanol

- Literature expectation: gas-phase ethanol has coupled methyl rotors and
  close trans/gauche conformers; trans/gauche torsional motion is known to be
  floppy and quantum-delocalized.
- nebwalk gives 11.73 kJ/mol, close to the rough conformer-scale target.
- Force disagreement strongly peaks at image 4, while energy disagreement is
  small and broad.
- Selected `(1, 4, 7)` is reasonable for the default force metric.

### Dimethyl Ether

- Literature expectation: dimethyl ether has coupled methyl internal rotors and
  low-energy torsional levels; literature spectroscopy reports torsional
  fundamentals/overtones in the few-hundred cm^-1 range.
- nebwalk gives 9.75 kJ/mol, close to the internal-rotor barrier scale.
- Both energy and force disagreement peak near image 4.
- Selected `(3, 4, 5)` is physically clean.

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

## Source Notes

- Ethane/propane conformational pattern and ethane 12.5 kJ/mol staggered vs
  eclipsed scale: Rotamer and eclipsed-conformation overviews.
  https://en.wikipedia.org/wiki/Rotamer
  https://en.wikipedia.org/wiki/Eclipsed_conformation
- Broader carbon-carbon torsional-barrier context: Nam et al.,
  "Explaining and Fixing DFT Failures for Torsional Barriers".
  https://arxiv.org/abs/2102.06842
- Methanol internal-rotor constant context: Jansen et al.,
  "Sensitivity of Transitions in Internal Rotor Molecules to a Possible
  Variation of the Proton-to-Electron Mass Ratio".
  https://arxiv.org/abs/1109.5076
- Dimethyl ether torsional spectroscopy context: Fernandez et al.,
  "New spectral characterization of dimethyl ether isotopologues...".
  https://arxiv.org/abs/1902.06444
- Ethanol conformer/torsional context: Nandi et al.,
  "Quantum calculations on a new CCSD(T) machine-learned PES reveal the leaky
  nature of gas-phase trans and gauche ethanol conformers".
  https://arxiv.org/abs/2206.02297
