# Scientific guidance

An optimizer reporting convergence is necessary but not sufficient evidence of
a physical transition pathway. For quantitative work:

1. Relax endpoints consistently and verify they are local minima.
2. Converge image count, force threshold, cell/slab size, k-points, cutoffs, and
   spin settings.
3. Inspect image spacing, geometry, forces, and the full energy profile.
4. Repeat suspicious paths with different interpolation or initial paths.
5. Refine the saddle and verify exactly one relevant imaginary mode.
6. Compare against an independent implementation, higher-level calculator, or
   experiment where the comparison is physically like-for-like.

MLIP accuracy is domain-dependent. Cross-model disagreement can identify model
discord but is not calibrated uncertainty and cannot prove that either model is
correct. DFT results remain dependent on functional, pseudopotentials, numerical
settings, and physical model choices.

Curated benchmark results should include structures, exact configuration,
software versions, machine-readable outputs, primary references, and checksum
or reproduction metadata.
