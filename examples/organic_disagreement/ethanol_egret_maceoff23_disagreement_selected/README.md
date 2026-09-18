# NEBwalk selected images

These images were selected from an MLIP-assisted NEB path using the `uncertainty_disagreement` strategy.

They are intended for DFT/QE refinement, single-point validation, or later labeling in an active-learning workflow.

This folder does not contain a complete DFT workflow by itself.

Selection used cross-model disagreement between two independently-trained MLIPs
as an uncertainty proxy, not a calibrated uncertainty quantification method (no
committee/ensemble was trained). This proxy is only meaningful where both
calculators are within their validated chemical domain; consult NEBwalk's
documented domain-failure list before trusting results outside that domain.
