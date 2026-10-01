# Proposed additions to the three Hugging Face model cards (NOT yet applied; needs the author)

Add to `healthmate-minilm-l6-v2-medical-3fold` and `-best2`:

> **Evaluation notice (2026-09-30).** Training data came from text extracted from a copyrighted
> medical encyclopedia; the training code and splits are not available. In zero-shot tests on
> BEIR SciFact, NFCorpus, TREC-COVID and a MedQuAD question-to-answer task, this model scores far
> below its base model all-MiniLM-L6-v2 (e.g. SciFact nDCG@10 0.249 vs 0.645). Its embeddings
> show a much lower effective rank (27.5 vs 138.5). We do not recommend it for retrieval.
> Earlier claims of Spearman 0.8039 / +18.31% cannot be reproduced and refer to an in-corpus
> similarity task. Not medical advice. Details and code: <research repo URL>.

Add to `healthmate-mistral-7b-medical-lora`:

> Training data are not documented in this card and have not been independently evaluated.
> Training logs show about 800 samples and a reported train loss near 12, which has not been
> explained. No generation or safety evaluation exists. Not medical advice.
