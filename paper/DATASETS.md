# Dataset and model licence register

Status: DRAFT. "HF card" = licence label on the Hugging Face page, read 2026-09-30
(summarised by a fetch tool; re-read the page itself before relying on it).
HF labels for BEIR mirrors can differ from the upstream source's terms, so every
BEIR row needs the upstream licence checked before we redistribute anything.

| Item | Use | Licence as read | Size | Open question |
|---|---|---|---|---|
| MedlinePlus health topics (NLM) | retrieval corpus | Public domain, attribution required. A.D.A.M. encyclopedia, ASHP drug monographs and most images are copyrighted | bulk XML | Must exclude copyrighted sub-collections |
| MedQuAD | consumer-health QA eval | CC BY 4.0. Answers for A.D.A.M., MedlinePlus drugs and herbal/supplements removed | 47,457 QA pairs | Some sources (e.g. cancer.gov) need their own terms checked |
| BEIR SciFact | retrieval eval | HF card: CC BY-SA 4.0 | 5,183 docs / 1,109 queries | Check upstream allenai/scifact terms |
| BEIR NFCorpus | retrieval eval | HF card: CC BY-SA 4.0 | 3,633 docs / 3,237 queries | Check upstream terms |
| BEIR TREC-COVID | retrieval eval | HF card: CC BY-SA 4.0. CORD-19 is CC0 with some publisher restrictions | 171,332 docs / 50 queries | Verify publisher restrictions |
| BAAI/bge-small-en-v1.5 | baseline | MIT | 33.4M params | none |
| intfloat/e5-small-v2 | baseline | MIT | 33.4M params | none |
| ncbi/MedCPT-Query-Encoder | baseline | Public domain (HF card); not for clinical decisions | - | Need article encoder too |
| all-MiniLM-L6-v2, GTE, BioLORD, PubMedBERT | baselines | NOT YET CHECKED | - | check |
| PubMedQA, MedQA, Mistral-7B-Instruct-v0.2 | generation eval / generator | NOT YET CHECKED | - | check |
| Gale Encyclopedia of Medicine | none | Copyrighted. Excluded from all new work | - | Existing HF weights trained on it: see PREREG section 7 |

Rule: we publish code, configs, IDs and derived statistics. We redistribute
no dataset text unless its licence above clearly allows it.
