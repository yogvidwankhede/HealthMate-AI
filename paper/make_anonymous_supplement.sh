#!/usr/bin/env bash
# Builds an anonymised copy of the research package for double-blind review (no git history, no author strings).
# Usage (repo root): ./paper/make_anonymous_supplement.sh /path/to/outdir
set -e
OUT=${1:-anon_supplement}
rm -rf "$OUT" && mkdir -p "$OUT"
cp -R paper "$OUT/paper"
rm -f "$OUT"/paper/CAMERA_READY_AUTHOR_BLOCK.md "$OUT"/paper/make_anonymous_supplement.sh "$OUT"/paper/results/*.log "$OUT/paper/tex/paper.pdf" "$OUT/paper/HUMAN_TODO.md" "$OUT/paper/MODEL_CARD_UPDATE.md" "$OUT/paper/REVIEW_SIMULATION.md" "$OUT/paper/download_mistral.py"
# replace the author's Hugging Face / GitHub names with ANON placeholders
grep -rlE "yogvid|Yogvid|wankhede|Wankhede" "$OUT" | while read f; do
  sed -i.bak -E 's#/Users/yogvid#/home/user#g; s#yogvidwankhede/healthmate#ANON/healthmate#g; s#yogvidwankhede#ANON#g; s#Yogvid( Vishwas)? Wankhede#ANON#g; s#wankhede#anon#gI' "$f" && rm -f "$f.bak"
done
# strip registry and archive links that identify the author
grep -rlE "osf\.io|zenodo|10\.5281|10\.17605|archive\.org/details/osf" "$OUT" | while read f; do
  sed -i.bak -E 's#https?://(doi\.org/)?(osf\.io|zenodo\.org)[^ )]*#[link removed for review]#g; s#https://archive\.org/details/osf-registrations-[a-z0-9-]+#[link removed for review]#g; s#10\.5281/zenodo\.[0-9]+#[DOI removed for review]#g; s#10\.17605/OSF\.IO/[A-Za-z0-9]+#[DOI removed for review]#g' "$f" && rm -f "$f.bak"
done
echo "Remaining identifying strings:"; grep -rniE "yogvid|wankhede|wustl|gmail|upwork|orcid|leonardo|osf\.io|zenodo" "$OUT" | cut -c1-160 | head -10 || true
(cd "$OUT/.." && zip -qr "$(basename $OUT).zip" "$(basename $OUT)")
echo "built $OUT and $OUT.zip"
