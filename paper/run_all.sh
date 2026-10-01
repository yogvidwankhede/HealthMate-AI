#!/usr/bin/env bash
# Regenerates every result. From the repo root, with the venv active and data/ populated:
#   data/beir/{scifact,nfcorpus,trec-covid} (BEIR zips), data/medquad (git clone abachaa/MedQuAD),
#   data/mplus (MedlinePlus topics XML, 2026-09-30). Skips outputs that already exist.
set -e
python paper/prep_medquad.py
for s in 13 42 2024; do
  [ -d models/ft_s$s ] || python paper/train_finetune.py --lr 5e-5 --epochs 3 --seed $s --out models/ft_s$s
done
[ -d models/ft_avg ] || python paper/average_models.py models/ft_avg models/ft_s13 models/ft_s42 models/ft_s2024
SYSTEMS="bm25 minilm-base bge-small e5-small gte-small medcpt hm-3fold hm-best2 path:models/ft_s13 path:models/ft_s42 path:models/ft_s2024 path:models/ft_avg hybrid:bge-small hybrid:path:models/ft_avg"
for d in scifact nfcorpus medquad trec-covid; do
  for m in $SYSTEMS; do
    t=$(echo "$m" | tr ':/' '--')
    [ -f paper/results/${d}__$t.json ] || python paper/eval_retrieval.py --dataset $d --model $m
  done
done
python paper/analyze.py
