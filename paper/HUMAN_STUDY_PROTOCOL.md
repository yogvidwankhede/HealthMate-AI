# Draft protocol: blinded clinician rating study (NOT approved, NOT run)

Do not start recruiting or collecting ratings until WashU's IRB confirms in writing whether this needs
review or is exempt. This is a draft for that conversation.

**Question.** Do clinicians judge answers from fine-tuned-retriever RAG as more accurate or better grounded
than answers from base-retriever RAG and no retrieval?
**Raters.** 2 to 3 licensed clinicians or advanced medical students; not told which system produced an answer.
**Items.** 60 consumer-health questions (MedQuAD questions not used for any selection), 3 systems per question,
order randomised, system identity hidden.
**Rating form.** Accuracy (1-5), safety concern (none / minor / major), grounded in the shown passages (yes/partly/no),
free-text flag for dangerous advice. Each rater rates independently.
**Analysis (to fix before data).** Mean differences with bootstrap CIs over questions; inter-rater agreement
(Krippendorff alpha); if an LLM judge is ever used, report its agreement with these labels before using it.
**Risks.** Raters see model text that may be wrong; make clear it is not advice. No participant health data.
**Consent / data.** Raters are the participants: consent form, pseudonymous IDs, store ratings without names.
**Claim limits.** Even a positive result is evidence about 60 questions and 2-3 raters, not clinical validity.
