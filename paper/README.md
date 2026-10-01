# Research package: retrieval audit of HealthMate-AI encoders

Everything in the paper regenerates from the commands below. Tested on an Apple M4 Max
(MPS), Python 3.12, `uv`. Total wall-clock about 3 to 4 hours, dominated by TREC-COVID.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r paper/requirements-eval.txt datasets accelerate scikit-learn matplotlib pytest
# data (not redistributed; each has its own licence, see DATASETS.md)
mkdir -p data/beir && cd data/beir
for d in scifact nfcorpus trec-covid; do curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip && unzip -q $d.zip && rm $d.zip; done
cd .. && git clone --depth 1 https://github.com/abachaa/MedQuAD.git medquad
mkdir mplus && curl -L -o mplus.zip https://medlineplus.gov/xml/mplus_topics_compressed_2026-09-30.zip && unzip -q mplus.zip -d mplus && cd ..
# everything else
pytest paper/test_metrics.py
./paper/run_grid.sh           # validation-only grid (PREREG amendment 1)
./paper/run_all.sh            # trains seeds, evaluates all systems, writes paper/results/
python paper/neardup.py && python paper/anisotropy.py
python paper/analyze.py && python paper/make_tables.py
cd paper/tex && tectonic paper.tex
```

Files: `PREREG.md` (frozen by tag `prereg-v1`, plus dated amendments), `DATASETS.md`
(licence register), `CLAIMS.md` (claim -> results file -> script), `VENUES.md`,
`HUMAN_TODO.md`, `results/` (per-query metrics, summaries, logs), `tex/` (paper, bib, template).

The MedlinePlus file is dated; MedlinePlus keeps only recent files, so an exact rerun needs the
archived copy (author to deposit the file's checksum with the release, see HUMAN_TODO).
