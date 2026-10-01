"""Near-duplicate check (PREREG amendment 1): MedQuAD answers vs MedlinePlus training chunks.
Writes paper/results/neardup.json with counts above thresholds and the flagged query ids."""
import json, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, "paper")
from train_finetune import load_topics, is_val
T = [t for t in load_topics() if not is_val(t["id"])]
train = [c for t in T for c in t["chunks"]]
corpus = [json.loads(l) for l in open("data/beir/medquad/corpus.jsonl")]
vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2).fit(train + [d["text"] for d in corpus])
Tm = vec.transform(train)
mx = np.zeros(len(corpus))
for i in range(0, len(corpus), 500):
    A = vec.transform([d["text"] for d in corpus[i:i + 500]])
    mx[i:i + 500] = (A @ Tm.T).max(axis=1).toarray().ravel()
flag = {d["_id"] for d, m in zip(corpus, mx) if m > 0.8}
qrels = [l.split("\t") for l in open("data/beir/medquad/qrels/test.tsv").read().splitlines()[1:]]
fq = sorted({q for q, d, _ in qrels if d in flag})
out = {"n_answers": len(corpus), "pct_max_cos": {str(p): float(np.percentile(mx, p)) for p in (50, 90, 99, 100)},
       "n_answers_over_0.8": len(flag), "n_answers_over_0.6": int((mx > 0.6).sum()), "flagged_query_ids": fq}
json.dump(out, open("paper/results/neardup.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "flagged_query_ids"}, len(fq))
