#!/usr/bin/env bash
# PREREG amendment 3 grid. From repo root, venv active, data/pubmedqa.jsonl and data/rag/ built.
set -e
for g in base lora42; do for r in none base ours hm3fold; do
  [ -f paper/results/rag/${g}__${r}.json ] || python paper/rag_eval.py --gen $g --ret $r
done; done
for g in lora123 lora999; do for r in none base; do
  [ -f paper/results/rag/${g}__${r}.json ] || python paper/rag_eval.py --gen $g --ret $r
done; done
python paper/rag_analyze.py
