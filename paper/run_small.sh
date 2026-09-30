#!/usr/bin/env bash
# Regenerates results for the two small BEIR sets. Run from the repo root.
set -e
for d in scifact nfcorpus; do
  for m in bm25 minilm-base bge-small e5-small hm-3fold hm-best2; do
    python paper/eval_retrieval.py --dataset $d --model $m
  done
done
