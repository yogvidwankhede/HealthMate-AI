# Changelog

## [0.1.0] - 2026-10-05
First tagged release. It records the state of the research package; it does not make the original chatbot's claims verified.

### Added
- `paper/`: retrieval audit with preregistration (seven dated amendments), dataset licence register, evaluation and training scripts,
  per-query results, analysis, claim checker (38 checks), ACL-format paper draft, and an anonymised-supplement builder.
- Community files, issue and PR templates, Dependabot, CI for the metric tests, OSS audit (`docs/OSS_AUDIT.md`).

### Changed
- README corrected to match the code and results files, with a status notice.

### Removed
- `pdf_cache.pkl` (extracted text of a copyrighted book) removed from the repository and its history.

### Known problems
- The original fine-tuning code, notebooks and question files are not in the repository; the headline numbers in older documents cannot be reproduced.
- The published LoRA adapters contain NaN weights and cannot be used.
- Eight Dependabot pull requests for old pinned dependencies are untested.
