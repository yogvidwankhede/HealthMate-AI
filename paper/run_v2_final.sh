#!/usr/bin/env bash
# PREREG amendment 6: final models for the setting selected on the DEV split (recipe r2, lr 5e-6, 1 epoch), then all benchmarks.
set -e
for s in 13 42 2024; do
  [ -d models/v2final_s$s ] || python paper/train_v2.py --recipe r2 --lr 5e-6 --epochs 1 --seed $s --out models/v2final_s$s
done
[ -d models/v2final_avg ] || python paper/average_models.py models/v2final_avg models/v2final_s13 models/v2final_s42 models/v2final_s2024
for d in scifact nfcorpus medquad trec-covid; do
  for m in s13 s42 s2024 avg; do
    [ -f paper/results/${d}__v2final_$m.json ] || python paper/eval_retrieval.py --dataset $d --model path:models/v2final_$m --tag v2final_$m
  done
done
./paper/run_rag2.sh path:models/v2final_avg
