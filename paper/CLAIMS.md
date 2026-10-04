# Claims ledger (results commit 5478083)

Automated: `python paper/check_claims.py` asserts every number quoted in the paper prose against the results files (38 checks, all pass at 5478083).

| Claim in paper | Results file | Script |
|---|---|---|
| nDCG@10 and CIs in Table 1 / Figure 1 | paper/results/<dataset>__<system>.json | eval_retrieval.py, make_tables.py |
| Paired differences and Holm p (Table 3) | paper/results/paired.json, summary.md | analyze.py |
| Published checkpoints far below base on 4 sets | same | eval_retrieval.py |
| Effective rank 27.5 vs 138.5; mean cosines | paper/results/anisotropy.json | anisotropy.py |
| Ours below base on all four; clear on 3, not TREC-COVID | paired.json | analyze.py |
| Seed range small vs gap to base | per-system JSONs (seed models) | make_tables.py |
| 4 of 14,798 MedQuAD answers have TF-IDF cosine > 0.8 to training chunks | results/neardup.json | neardup.py |
| Validation grid; best = lr 5e-5, 3 epochs; base val MRR 0.997 | results/train/*.json | run_grid.sh, train_finetune.py |
| Dataset sizes (queries, docs) | per-system JSONs (n_queries, n_docs) | eval_retrieval.py |
| 919/93 topics, 6,666 pairs | results/train/*.json | train_finetune.py |

| All 448 tensors of each published LoRA adapter are NaN | results/adapter_nan.json | check_adapters.py |
| RAG on PubMedQA: base Mistral, accuracy 0.150-0.162 in all retrieval conditions; no paired difference excludes zero | results/rag/*.json, rag_summary.md | rag_eval.py, rag_analyze.py |
| LoRA runs invalid (NaN logits), excluded | results/rag/lora*.json, rag_summary.md | rag_analyze.py |

| MedQuAD at 512 tokens: base 0.591, ours 0.529, published 0.164; gap -0.063 (CI excludes 0) | results/seq512.json | eval_retrieval.py --max-seq 512, analyze.py |
| Stronger recipe R2: dev grid (all 12 below base 0.693; best 0.659), final test-split differences vs base (-0.006, -0.005, -0.017, -0.023) | results/medquad__dev__*.json, results/v2_summary.json | run_v2_grid.sh, run_v2_final.sh, analyze_v2.py |
| RAG v2: accuracy and hit@1 per retriever; gold abstract 0.803 vs none 0.631; published encoder -0.061 vs base retriever (exploratory) | results/rag2/*.json, rag2_summary.md, rag2_paired.json | rag2_eval.py, rag2_analyze.py |

Non-empirical statements and their sources:
- Training code, notebooks, question CSVs are absent from the repo: file listing and git history of yogvidwankhede/HealthMate-AI at 436343f (pre-rewrite) and 6b12e35.
- Headline 0.8039 vs 0.6795: README and HF model cards. The alternative 0.7552 vs 0.6768 is from the course report "HealthMate AI Main.pdf" in the author's local folder (not in any repo); verify before citing.
- Training pairs were retrieved chunk + reference answer filtered by BLEU/ROUGE-L: same course report.
- Licences: paper/DATASETS.md (HF/GitHub pages read 2026-09-30).

Not claimed (no evidence): hallucination, faithfulness, answer quality, clinical value, why the published models collapsed.
