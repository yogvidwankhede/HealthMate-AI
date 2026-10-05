# ACL Responsible NLP Research checklist: draft answers (items read from aclrollingreview.org on 2026-10-04)
The submission system's version is the one that counts; copy answers there and correct anything that has changed.

A1 limitations: Yes (Limitations section). A2 risks: Yes (Ethics Statement; models not for clinical use).
B1 cite creators: Yes. B2 licences/terms: Yes in the repo (paper/DATASETS.md); the paper states public-domain/CC BY terms in the Ethics Statement only briefly, so add one sentence per dataset if space allows.
B3 intended use: Yes (Data checks paragraph). B4 PII/offensive checks: Partly. The paper says no check was done exhaustively; answer "No" unless you add one.
B5 documentation (domain, language, demographics): Partly. English biomedical text; no demographics. Answer honestly.
B6 statistics (examples, splits): Yes (sizes of all sets, 919/93 topic split).
C1 parameters, budget, infrastructure: Yes (Compute and software). C2 setup and hyperparameter search: Yes (grid and best values). C3 descriptive statistics, single vs aggregate: Yes (seeds reported separately and averaged; CIs).
C4 package details: Yes (versions).
D1 to D5 human participants: N/A (no annotators or participants).
E1 AI assistants: Yes (LLM Use Disclosure paragraph).
