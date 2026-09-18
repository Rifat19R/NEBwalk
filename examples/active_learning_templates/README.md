# Active-learning production templates

`mace_qe_campaign.json` is a configuration skeleton, not a validated physical
calculation. Replace every placeholder and converge the calculator settings for
the system. The MXene and vacancy files list domain-specific checks that must be
resolved before adapting the base template.

Run `NEBwalk mlip check` before campaign initialization. Validate endpoint
structures independently. Sparse QE labels do not constitute a full QE NEB and
must not be reported as a DFT barrier.
