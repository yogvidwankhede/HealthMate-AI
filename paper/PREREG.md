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
Amendment 1 (2026-09-30, before any fine-tuned model of ours was trained or evaluated;
the zero-shot baseline and old-checkpoint runs had already been done):
- MedQuAD is used as a question -> answer retrieval set built by paper/prep_medquad.py.
  Sources: CancerGov, GARD, GHR, NIDDK, NINDS, SeniorHealth, NHLBI, CDC.
  Excluded: MedlinePlus Health Topics (it overlaps the MedlinePlus training corpus) and the
  three sub-collections whose answers were removed for copyright.
- Near-duplicate check: each MedQuAD answer is compared with every training chunk by TF-IDF
  cosine; counts above 0.8 are reported, and results are also reported with those answers'
  queries removed.
- Fine-tuning data: MedlinePlus English health topics (2026-09-30 XML). Each full summary is
  cut into chunks of at most 120 words on sentence boundaries. Pairs: (topic title, chunk)
  and (meta description, chunk). Loss: MultipleNegativesRankingLoss with no duplicate anchors
  per batch. Split by topic: 90% train / 10% validation, fixed hash of the topic id.
- Model selection on validation only (validation = title/meta-description -> own chunks, MRR@10):
  lr in {1e-5, 2e-5, 5e-5}, epochs in {1, 2, 3}. Final models: 3 seeds (13, 42, 2024) plus
  a uniform weight average of the 3 seed models. Benchmarks are never used for selection.
- Correction: the MedCPT licence is "public-domain" (licence_name on the HF card), not plain
  "other". BioLORD-2023 needs UMLS/SNOMED licensing by the user, so it is excluded unless the
  author confirms a UMLS licence.
- Hybrid = reciprocal rank fusion (k=60) of BM25 and one dense model over their top-100 lists.
- BM25 uses rank-bm25 (Okapi, default k1/b), not Anserini; absolute BM25 numbers differ slightly from BEIR's.

Amendment 2 (2026-09-30, after the validation grid, before any benchmark run on our models):
- Selection rule applied as preregistered (highest validation MRR@10, seed 13): lr 5e-5, 3 epochs
  (val MRR 1.000). This is the edge of the grid; the grid was not extended. Base MiniLM scores
  0.997 on this validation task, so the validation task is saturated and weakly informative; this is
  reported as a limitation. Final models: seeds 13, 42, 2024 and their uniform weight average.

Amendment 3 (2026-10-01, before any generation run; retrieval results were already known):
Small local RAG study, answer correctness only. Hardware: M4 Max, 38.6 GB; no cloud spend.
- Task: PubMedQA expert-labelled set (pqa_labeled, MIT licence on its HF card), 500 questions drawn
  with seed 13. Labels yes/no/maybe. The question is given WITHOUT the abstract; retrieved MedlinePlus
  chunks are the only optional context, so RAG can only help via general consumer-health text.
- Prediction by next-token scoring (no sampling): the prompt ends with 'Answer (yes, no or maybe):' and
  the label is the highest-scoring of the three first tokens. Metrics: accuracy and macro-F1, bootstrap
  95% CIs over questions, paired bootstrap vs the no-retrieval condition of the same generator.
- Retrieval corpus: all MedlinePlus English topic chunks (120 words), top-3 by cosine, from one of:
  base MiniLM, our seed-averaged model, published HealthMate 3-fold. Condition 'none' uses no context.
- Generators: Mistral-7B-Instruct-v0.2 (fp16) and the three published LoRA adapters (seeds 42, 123, 999).
  Full grid (4 retrievers x base and adapter-42) plus adapters 123 and 999 with the none and base-MiniLM retrievers.
- NOT measured: faithfulness, hallucination, answer fluency, safety/refusal behaviour. No LLM judge is used,
  because none has been calibrated against human labels. Any such claim is out of scope.
- Known weakness: the corpus is unlikely to contain the answer to a PubMedQA question, so a null result
  is expected and would not show that retrieval helps or hurts in a better-matched setting.

Amendment 4 (2026-10-01, after the first run, base generator with no retrieval, showed 92% 'maybe' predictions
and accuracy 0.156; that run was discarded and re-run with scores saved):
- The primary analysis stays as preregistered (argmax over yes/no/maybe).
- Secondary analysis, added after seeing this collapse and therefore exploratory: yes-vs-no accuracy on the
  questions whose label is yes or no, choosing the higher of the 'yes' and 'no' scores. It removes the
  generator's tendency to hedge from the comparison.

Amendment 5 (2026-10-04, exploratory, after seeing all main results; prompted by the simulated review): MedQuAD was re-run for
the base, published 3-fold and our averaged model with max_seq_length 512 (the default for MiniLM variants is 256), to check whether
truncation drives the gaps. Output files carry the tag '-seq512'. This is a post hoc robustness check, not a new primary result.

Amendment 6 (2026-10-04, written BEFORE any run of the recipes below; all main results above were already known, so this is a
follow-up study, not a fresh preregistration): a stronger re-training baseline, prompted by the simulated review.
- Question: does a better-designed fine-tuning recipe on the same MedlinePlus data beat the base model?
- Recipes (all start from all-MiniLM-L6-v2, same topic split, sentence-transformers trainer, MultipleNegativesRankingLoss):
  R1 = (anchor, chunk, hard negative): negative is a chunk of a DIFFERENT topic ranked 5 to 30 by the base model for that anchor,
  one drawn at random with a fixed seed. R2 = R1 plus replay of general question-answer pairs (sentence-transformers/natural-questions,
  an equal number of pairs, sampled with seed 13; its licence was not confirmed, so it is used locally and never redistributed).
- Grid: lr in {5e-6, 1e-5, 2e-5}, epochs in {1, 2}, seed 13, for each recipe (12 runs).
- Selection data (new, because the MedlinePlus validation task is saturated): MedQuAD sources GARD and GHR are the DEV split;
  nDCG@10 on their queries (corpus = all 14,798 answers) picks the recipe and setting. Final report uses the TEST split = CancerGov,
  NIDDK, NINDS, SeniorHealth, NHLBI and CDC queries, plus SciFact, NFCorpus and TREC-COVID, none of which are used for selection.
  Baselines and the earlier models are re-scored on the same DEV/TEST split from their saved per-query results (no re-run needed).
- Final models: the selected setting with seeds 13, 42, 2024 and their uniform weight average.
- Primary test: paired bootstrap of the averaged model against base MiniLM on each test set; Holm over these four comparisons.
- If no recipe beats base, that is the result.
