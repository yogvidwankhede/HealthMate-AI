# Adversarial review and reproducibility audit (simulated, 2026-10-04)

Target form: ACL Rolling Review. Fields found on the live guidelines page: soundness, excitement, reproducibility,
overall assessment, confidence. The exact 1-5 scale definitions were not retrievable, so scores below are my own judgement on
a 1-5 scale and not calibrated to ARR's wording.

## Step 1. Reviewer report
**Summary.** The paper audits sentence encoders released by a student RAG project. The released checkpoints score far below
their base model on four external retrieval sets. A re-training on public-domain MedlinePlus text also scores below base. A small
PubMedQA test finds retrieval has no effect on a 7B generator, and the published LoRA adapters are NaN.

**Strengths.** Claims trace to code and results (25 automated checks). Negative results are reported with CIs and Holm-corrected
tests. The NaN-adapter finding is verified and useful. Licences are documented.

**Weaknesses.**
1. Narrow, low-novelty contribution: an audit of one unpublished student project. Little transfers beyond "check your
   fine-tuned model on external data".
2. The re-training is a weak test of "domain fine-tuning". 1,012 topics, an easy title-to-chunk proxy task (the base model
   already scores 0.997 validation MRR@10), a best setting at the edge of the grid, no hard negatives, no general-data
   replay, no learning-rate warm-up search. A negative result here says little about adaptation methods that work.
3. The paper cannot say why the published encoders collapsed. The training code and data are missing, and the paper
   admits it. "Audit" of a black box with no mechanism.
4. "Preregistered" is overstated: the registry is a git tag by the same author, and baselines and the old-checkpoint runs
   preceded it. Four amendments were added after results (some, like the yes/no analysis, post hoc).
5. The MedQuAD task is built by the authors from template questions ("What is (are) X?"). Answers are truncated to 256
   tokens for MiniLM models and 512 for others, which favours the longer-context baselines. The conclusion that ours is
   below base on MedQuAD may partly reflect this.
6. The RAG study is weak. The base generator answers "maybe" 90% of the time, so accuracy (0.16) mostly measures the prompt,
   not retrieval. The retrieval corpus cannot answer PubMedQA questions. No claim about RAG follows from it.
7. Statistics: 50 queries for TREC-COVID; bootstrap is over queries only, not over training seeds; the comparison against the
   best baseline per dataset is chosen post hoc; BM25 is a non-standard implementation.
8. Missing baselines: BioLORD, PubMedBERT-based retrievers, larger BGE/E5, rerankers, and domain-adaptation methods
   (GPL, TSDAE are cited but not run). No consumer-health query set with real user questions.

**Questions for the authors.** Is the published model's collapse reproducible from the original recipe? What do the base and
re-trained models score on the original in-corpus task? Does the gap shrink with 256 vs 512 token truncation matched across
models? Why "maybe" in every condition and does a different prompt change the picture?

**Missing related work (verified to exist, not compared).** GPL (arXiv 2112.07577) and TSDAE (arXiv 2104.06979): now cited.
Others I have not verified and so do not list.

**Scores (my judgement, 1-5).** Soundness 3. Excitement 2. Reproducibility 4. Confidence 3. Overall: borderline for a
workshop, below the bar for a main track.

**What would change my recommendation.** A recovered original recipe that reproduces the collapse, or a stronger
re-training (hard negatives, replay, larger corpus) that also fails, or a real consumer-health query set.

**Reviewer 2 (harsh).** This is a course project post-mortem dressed as a paper. The authors found that their own earlier
checkpoint is bad and that a small, weak re-training is also bad, and conclude that "self-supervised fine-tuning did not help".
They cannot explain the failure, tested one recipe, built their own benchmark, and ran a 500-question RAG test where the model
says "maybe" nine times in ten. The "preregistration" is a git tag. Reject, or at most accept as a short workshop note.

## Step 2. Claims audit
All numerical claims map to results files in `paper/CLAIMS.md`. `python paper/check_claims.py` asserts them (25 pass).
UNSUPPORTED or weakly supported in prose: "pseudo-labelled in-corpus training is one candidate" (hypothesis, flagged as such);
the course-report discrepancy (0.8039 vs 0.7552) rests on a PDF outside the repo.

## Step 3. Statistical audit
- Leakage: MedQuAD/MedlinePlus overlap mitigated (MedlinePlus source excluded; 4 of 14,798 answers above 0.8 TF-IDF cosine).
  Not checked: semantic (embedding) near-duplicates; leakage between benchmark corpora and the pre-training of BGE/E5/GTE/MedCPT
  (unknowable; BEIR sets are widely used).
- Tuning: no tuning on benchmarks. Validation grid used a saturated task.
- Multiple comparisons: Holm applied across the comparisons in the analysis script; the "ours vs best baseline" rows are post hoc.
- CIs: present for all systems. Seeds: three; no seed-level inference.
- Wording: causal language avoided ("did not establish the cause"). The title asks a question.
- Cherry-picking: no qualitative examples shown.

## Step 4. Reproduction test (what I actually ran)
Clean clone of `main` (commit b48ffd4) in a fresh venv with pinned requirements: unit tests pass (3); re-running SciFact base-MiniLM
evaluation reproduces the committed per-query metrics exactly (max abs diff 0.0); `analyze.py` + `make_tables.py` regenerate the three
table files identically. NOT re-run: training, TREC-COVID, MedCPT, the RAG grid (hours, large downloads). Known breakage risk:
the MedlinePlus XML is dated and may disappear; unpinned data URLs.

## Step 5. Citation audit
Seventeen references checked for existence against arXiv/Crossref (titles, first authors, years); two more added today (GPL, TSDAE).
Not done: reading each paper to confirm the sentence citing it. The statement "BGE is described in C-Pack" is standard but the
arXiv title is C-Pack; I did not verify the paper's text. No "first"/"novel" claims appear in the paper.

## Step 6. Ethics, licensing, anonymity
Datasets and models: see DATASETS.md. The audited Hugging Face weights were trained on copyrighted text; you chose to keep them
public (a live risk). Anonymity: the PDF is anonymous, but the repository, model cards and `setup.py` identify the author; a double-blind
submission needs an anonymised mirror or a link-free paper. AI disclosure paragraph present; ARR allows AI assistance with disclosure.
Human subjects: none in this paper.

## Step 7. Five hardest questions (draft answers) and one-week experiments
1. Why believe the collapse is not your setup? -> Same pipeline gives expected numbers for base/BGE/E5/GTE; geometry diagnostic
   (rank 27.5 vs 138.5). Experiment: evaluate with the other published pooling/normalisation variants.
2. Why is your re-training a fair test? -> It is not a general claim; one recipe. Experiment: add hard negatives, replay of general
   pairs, lower learning rate, and a 10x larger corpus (e.g. PubMed abstracts), prereg first.
3. Is MedQuAD truncation driving results? -> Experiment: rerun all models at 512 tokens and with chunked answers.
4. Does the RAG result mean anything? -> Probably not. Experiment: use a corpus that can answer the questions (PubMed abstracts) and
   an instruction-calibrated prompt, or drop the section.
5. Is the preregistration real? -> It is a git tag plus amendments. Experiment: deposit the final protocol on OSF before any new run.

## Recommendation: FIX-FIRST (not submission-ready)
Priority: (1) pick the venue format and trim to a workshop short or full paper; (2) either strengthen or drop the RAG section;
(3) matched-truncation MedQuAD rerun; (4) a stronger re-training baseline; (5) anonymised artefact link; (6) author-only items in
HUMAN_TODO.md; (7) read-through of every cited paper for the sentence it supports.

## Update 2026-10-04: citation read-through
Abstracts read for C-Pack (BGE English models are released with it), model soups (averaging improves accuracy and robustness; the
draft wrongly said "reduce variance", now fixed), Smart Reply (in-batch negatives confirmed in the text), and EWC (about sequential-task
forgetting; the draft's wording was too general, now fixed). Titles and first authors of all other references were checked against
arXiv/Crossref; I did not read their full text.

## Re-assessment after the follow-up work (2026-10-04)
Addressed: (2) stronger re-training baseline with a held-out dev split (result: matches base within noise, does not exceed it); (3) matched-truncation
MedQuAD check; (4) RAG redesigned so retrieval can matter (oracle beats no-context by 0.17); citation wording fixed; anonymised supplement builder added.
Still open and honest: the contribution remains an audit plus null results (low novelty); the preregistration is not on an external registry; the
original training recipe is unavailable so the cause of the collapse is untested; only one replay corpus with unconfirmed licence; no human or
faithfulness evaluation; each cited paper's full text was not read. Revised scores (my judgement): soundness 3.5, excitement 2, reproducibility 4,
confidence 3. Recommendation: **suitable for a workshop or Findings-style submission after the author-only steps; still not a main-track paper.**
Submission-blocking items are now author-only: authorship/IP, venue call confirmation, GitHub purge, anonymised link, ORCID/funding, final read-through.
