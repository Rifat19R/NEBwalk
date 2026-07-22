"""Pt vacancy migration in FCC Pt with EMT.

Reference barrier: ~1.49 eV DFT-PBE. Calculator: EMT.

EMT does not capture relativistic effects, so Pt is a known, documented
failure case (large error) rather than a nebwalk bug — kept here precisely
because it is instructive: a big EMT/DFT gap on Pt says "swap the
calculator," not "the NEB kernel is broken."
"""

from vacancy_benchmark_suite import main

if __name__ == "__main__":
    main("pt", "emt")
