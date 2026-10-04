#!/usr/bin/env bash
# PREREG amendment 7. Extra args are additional retrievers, e.g. path:models/v2_best. From repo root, venv active.
set -e
for r in none oracle base hm3fold ours bge "$@"; do
  n=$(echo "$r" | tr ':/' '-_'); [ -f paper/results/rag2/$n.json ] || python paper/rag2_eval.py --ret $r
done
