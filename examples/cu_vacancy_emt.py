"""Cu vacancy migration in FCC Cu with EMT.

Reference barrier: ~0.70 eV DFT-PBE. Calculator: EMT.
EMT is known to underestimate Cu vacancy barriers (~0.50 eV expected).
"""

from vacancy_benchmark_suite import main

if __name__ == "__main__":
    main("cu", "emt")
