#!/usr/bin/env bash
# Validation-only grid (seed 13), PREREG amendment 1. Run from repo root.
set -e
for lr in 1e-5 2e-5 5e-5; do for ep in 1 2 3; do
  python paper/train_finetune.py --lr $lr --epochs $ep --seed 13 --out models/grid_lr${lr}_e${ep}
done; done
