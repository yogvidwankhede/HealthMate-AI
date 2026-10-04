# OSS readiness audit: HealthMate-AI only (2026-09-30)

## 1. Eligibility facts for Claude for Open Source
Program page fetched live 2026-09-30 (https://claude.com/contact-sales/claude-for-oss). Five routes.

| Route | Threshold | This repo | Verdict |
|---|---|---|---|
| Maintainer / library author | 500 dependent repos, 100 dependent packages, or 200k monthly downloads | No package published (a `setup.py` exists, nothing on PyPI); dependents not measurable and effectively none | Clearly not met |
| Core contributor | listed maintainer of a recognised foundation project | n/a | Not applicable |
| Active contributor | 100 merged PRs into others' repos in 12 months | Account-wide; out of scope for this audit | Not assessed |
| Community builder | 20 unique external contributors with merged PRs in 12 months | 0 merged PRs, 1 contributor (owner), 1 fork | Clearly not met |
| Critical infrastructure | OpenSSF criticality score >= 0.4 | tool not run; repo has 0 stars, 0 issues, 0 dependents, so expected near 0 | Clearly not met (not measured) |

Repo facts (GitHub API): public, Apache-2.0, 0 stars, 1 fork, 0 open issues, created 2025-12-15, 7 topics, 0 releases.
Applicants who miss thresholds may still apply under "something the ecosystem quietly depends on"; this repo
does not support that claim today.

## 2. Upstream contribution shortlist (proposals only, nothing opened)
Checked 2026-09-30: BEIR's README invites issues/PRs and states it does not vouch for dataset licences; no AI
policy found in it. sentence-transformers: no CONTRIBUTING file at the standard path (policy not verified).

| Project | Candidate | Evidence | Wanted? |
|---|---|---|---|
| MedQuAD (abachaa/MedQuAD) | Issue: question ids are not unique across source folders | Parsing all 12 folders gives 47,441 QA pairs but 30,086 distinct `qid` values; 9,654 qids appear in more than one source folder (e.g. `0000559-1` in five). Anyone indexing by qid silently collides (this broke my first conversion script) | Plausibly; repo has few signs of activity, so unknown |
| BEIR | Issue/PR: note the HF-mirror licence labels (CC BY-SA 4.0) differ from some upstream sources' terms | Needs checking per dataset before filing; I have not verified the upstream terms | Unverified |
| Own models (HF) | Correct the model cards | See paper/MODEL_CARD_UPDATE.md | Yes, but it is your own repo |

Honest estimate: these are small documentation issues. They will not by themselves move this repo toward any
program threshold, and I make no numeric prediction. Disclose AI assistance in any upstream submission.

## 3. Badge map (GitHub profile achievements)
I could not retrieve GitHub's own achievements page (docs URL returned 404); the list below comes from third-party summaries
of the badges and must be re-checked on GitHub before you rely on it.
- Pull Shark (merged PRs): a normal branch-and-PR workflow on this repo feeds it. Fine to earn, not worth chasing.
- Quickdraw, YOLO (merge without review): quirks of speed and skipping review; do not chase, and YOLO conflicts with good practice.
- Starstruck (stars): needs a real audience; only a useful README, demos and a paper link help.
- Galaxy Brain (accepted discussion answers): depends on others' acceptance; help people genuinely, do not farm.
- Arctic Code Vault: no longer obtainable.
Gameable: Quickdraw, YOLO, Pull Shark via trivial self-merged PRs, Starstruck via star swaps. Do none of these.

## 4. Credibility changes (branch `docs/readme-accuracy`, local, not pushed)
README corrected; CONTRIBUTING, SECURITY, CODE_OF_CONDUCT (links to Contributor Covenant 2.1), PR and issue templates,
Dependabot, CITATION.cff (repo only; add the preprint when it exists), draft starter issues. Research branch adds a CI
workflow that runs the metric unit tests. Not done: releases/tags with changelog, Zenodo DOI, OpenSSF Scorecard run,
Best Practices badge, coverage badge, topics review. No badge claims anything not earned.

## 5. Docs vs code
README numbers vs results files: only BLEU/ROUGE-L figures trace (to `evaluation_summary_LATEST.json`); all embedding
numbers are untraceable; 16 listed files are missing; dependency versions in the README disagreed with requirements.txt
(fixed). Open issues: none exist.

## Out of scope, noted only
Account-wide contribution history; the Apara-App org; other repositories.

## 6. OpenSSF Scorecard (run 2026-10-01 on `main`, scorecard from Homebrew)
Aggregate **2.0 / 10**. Per check: Binary-Artifacts 10, Contributors 10, License 10, Maintained 1, and 0 for
Branch-Protection, CII-Best-Practices, Code-Review, Dependency-Update-Tool, Fuzzing, SAST, Security-Policy, Vulnerabilities.
Expected to improve after merging these PRs: Security-Policy (SECURITY.md) and Dependency-Update-Tool (Dependabot).
Needs you: Branch-Protection (repo settings), Code-Review (a reviewed PR; do not self-approve for show),
CII Best Practices (apply for the badge honestly), SAST (add CodeQL if wanted). Vulnerabilities likely come from old pinned
dependencies; run `pip-audit` before bumping. No Scorecard badge is added because the score is not worth advertising.
