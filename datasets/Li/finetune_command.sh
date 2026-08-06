#!/usr/bin/env bash
# Reviewable starting point -- see
# nebwalk.finetune.generate_finetune_command/TRAINING_SET_DISCLOSURE
# before running.
mace_run_train \
    --name=li_vacancy_finetune \
    --foundation_model=medium \
    --train_file=datasets/Li/li_train.extxyz \
    --valid_fraction=0.1 \
    --energy_key=REF_energy \
    --forces_key=REF_forces \
    --E0s='{"3": -195.3745826274783}' \
    --device=cuda \
    --max_num_epochs=200 \
    --batch_size=1
