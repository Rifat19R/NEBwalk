"""Ag vacancy migration in FCC Ag with EMT.

Reference barrier: ~0.66 eV DFT-PBE. Calculator: EMT.
"""

from vacancy_benchmark_suite import main

if __name__ == "__main__":
    main("ag", "emt")
