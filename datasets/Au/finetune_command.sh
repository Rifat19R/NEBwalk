#!/usr/bin/env bash
# Reviewable starting point -- see
# NEBwalk.finetune.generate_finetune_command/TRAINING_SET_DISCLOSURE
# before running.
mace_run_train \
    --name=au_vacancy_finetune \
    --foundation_model=medium \
    --train_file=datasets/Au/au_train.extxyz \
    --valid_fraction=0.1 \
    --energy_key=REF_energy \
    --forces_key=REF_forces \
    --E0s='{"79": -3734.059699519572}' \
    --device=cuda \
    --max_num_epochs=200 \
    --batch_size=1
