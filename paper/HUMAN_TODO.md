# What the human author must do (nothing here was done on your behalf)

0. The published LoRA adapters are all-NaN (verified). Decide whether to delete or replace them on Hugging Face; the card now says they are unusable.
1. Decide authorship and affiliation. A course report and the README citation list a co-author;
   you said it is only you. The README citation is unchanged until you confirm.
2. Check WashU/course rules on releasing coursework-derived code and the paper (IP, course policy).
3. Contact GitHub Support to purge the old commit 436343f (copyrighted Gale text) and any fork
   copies; the history rewrite is already pushed.
4. Apply or reject the model-card edits in MODEL_CARD_UPDATE.md (Hugging Face login needed).
5. Pick a venue after reading its live CFP (see VENUES.md); fill the venue checklist and AI-disclosure
   wording; remove identifying strings for double-blind review (repository, model cards, this repo's
   setup.py author fields, Upwork link in the case-study PDF).
6. Submit an arXiv preprint if the venue allows it (needs your arXiv account and endorsement status).
7. Provide ORCID, funding/acknowledgements, and a statement that you verified every claim in CLAIMS.md.
8. Mint a Zenodo DOI (link the repo in Zenodo settings) and add it to CITATION.cff.
9. If you want clinician evaluation: ask WashU IRB whether it needs review or exemption, then recruit raters.
   None was done; the paper makes no clinical claim.
10. Archive the MedlinePlus XML you used (checksum in the release) so the fine-tuning set can be rebuilt.
11. Check whether you hold a UMLS licence if you want BioLORD added (excluded here).
12. Review the research branch diff and merge it yourself; nothing new has been pushed except the
    approved history rewrite.
