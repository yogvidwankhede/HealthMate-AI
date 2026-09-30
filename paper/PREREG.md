# Pre-registration (DRAFT, not yet frozen)

Nothing here has been run. This file is frozen by a git tag `prereg-v1` BEFORE
any final experiment. After the tag, changes go in an "Amendments" section with
a date and reason; the paper states what was decided before vs after results.

## 1. Question
Does fine-tuning a small retrieval encoder (all-MiniLM-L6-v2) on an open
consumer-health corpus improve retrieval beyond (a) its own base, and (b) strong
open baselines, on external benchmarks? Secondary: does retrieval quality change
RAG answer correctness and faithfulness?

## 2. Why the old result is not reused
The earlier Spearman 0.8039 has no surviving code or split. It is not reported as
a finding. Retraining is under this protocol only.

## 3. Data (see DATASETS.md)
- Training corpus: MedlinePlus public-domain health topics only.
- Fine-tuning pairs: built by a documented procedure (TBD and frozen before
  training: e.g. title/section to passage pairs; any LLM-generated queries are
  labelled as synthetic).
- Splits: by document, never by sentence or chunk. Near-duplicate filter across
  splits (threshold TBD, frozen, and logged with counts removed).
- Evaluation: BEIR SciFact, NFCorpus, TREC-COVID; MedQuAD held out entirely
  from training (questions and answers checked for overlap with the corpus).
- Seeds: 13, 42, 2024 for every trained model. No tuning on any eval set;
  hyperparameters are chosen on a validation split from the training corpus.

## 4. Systems
Fine-tuned MiniLM (single run, 3-seed, and weight-average ensemble of seeds),
base MiniLM, BGE-small, E5-small, GTE-small, MedCPT, BioLORD, BM25, and a
BM25+dense hybrid. The old HF checkpoints are evaluated as-is, as a labelled
extra row with their known training-data contamination risk stated.

## 5. Metrics and tests
Primary: nDCG@10. Secondary: Recall@10/100, MRR@10. 95% bootstrap CIs over
queries (10,000 resamples). Paired bootstrap / permutation test vs base MiniLM
and vs the best baseline per dataset; Holm correction across the comparisons
listed here. Effects reported with CIs, not only p-values.

## 6. RAG evaluation (secondary, contingent on budget approval)
Conditions: no retrieval; base-embedding RAG; fine-tuned-embedding RAG; base vs
LoRA generator. Open QA sets TBD after licence check. Claim-level faithfulness
uses an LLM judge only if calibrated on a human-labelled sample with reported
agreement; otherwise reported as unvalidated. No clinical-validity claim without
clinician raters and IRB status known.

## 7. Known limitations fixed in advance
- Earlier HF weights were trained on copyrighted Gale text; open question
  whether they should stay public (author decision).
- Small encoders may not beat BGE/E5/MedCPT. That null result will be reported.
- No human study unless raters and IRB answer exist.

## 8. Analyses decided AFTER results (exploratory, labelled as such)
None yet.

## Amendments
(none)
