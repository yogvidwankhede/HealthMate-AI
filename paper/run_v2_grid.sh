#!/usr/bin/env bash
# PREREG amendment 6: 12-run grid, seed 13, scored on the MedQuAD DEV sources only. From repo root, venv active.
set -e
DEV="2_GARD_QA,3_GHR_QA"
python paper/eval_retrieval.py --dataset medquad --model minilm-base --sources $DEV --tag dev__minilm-base
for r in r1 r2; do for lr in 5e-6 1e-5 2e-5; do for ep in 1 2; do
  out=models/v2_${r}_lr${lr}_e${ep}
  [ -d $out ] || python paper/train_v2.py --recipe $r --lr $lr --epochs $ep --seed 13 --out $out
  [ -f paper/results/medquad__dev__v2_${r}_lr${lr}_e${ep}.json ] || python paper/eval_retrieval.py --dataset medquad --model path:$out --sources $DEV --tag dev__v2_${r}_lr${lr}_e${ep}
done; done; done
