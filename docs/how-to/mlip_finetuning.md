# MACE fine-tuning

Install the tested MACE integration range separately from the core package:

```bash
pip install "NEBwalk[training]"
NEBwalk mlip check --protocol naive
```

The core package never imports PyTorch or MACE at import time. The backend uses
the installed official MACE command-line programs, passes an argument vector
without a shell, records help-text and environment fingerprints, and preserves
stdout, stderr, configuration, command, model checksum, and status manifests.

`FineTuningConfig` supports `naive`, `lora`, and `multihead_replay`. Support for
a non-naive protocol is accepted only when the installed MACE CLI advertises
the required flags. Put backend-specific switches in `extra_args` as explicit
flag/value pairs; they are validated and never interpolated into a shell.

```bash
NEBwalk mlip finetune split fine_tuning.json model-output
NEBwalk mlip evaluate model-output/model_manifest.json split/test.extxyz evaluation
```

Model activation is based on a configured validation metric, not training loss.
Committee members are ranked on the validation split before the selected model
is evaluated once on the held-out test split; these metric roles are stored
separately.
The local registry stores checksums and provenance. A successful process exit
is necessary but not sufficient: inspect held-out energy, force, pathway, and
barrier errors, and test the model only in its intended chemical domain.

MACE, accelerator drivers, and foundation-model files remain optional external
dependencies. Production GPU runs should be isolated and monitored by the job
scheduler used at the deployment site.

Environment probing defaults to a 60-second timeout and evaluation to 300
seconds. Training limits are deployment-specific and default to `None`; set
`training_timeout_seconds` on `MACEFineTuningBackend` for unattended jobs.
`KeyboardInterrupt` records an interrupted result before propagating.
`NEBwalk mlip check` records installed Torch metadata without importing Torch;
add `--probe-accelerator` only when CUDA initialization is known to be safe.
