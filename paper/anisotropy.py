"""Diagnostic: mean cosine similarity between random passage pairs (higher = more compressed space).
Uses 2,000 random SciFact passages, seed 13. Writes paper/results/anisotropy.json."""
import json, random, numpy as np
from sentence_transformers import SentenceTransformer
random.seed(13)
docs = [json.loads(l) for l in open("data/beir/scifact/corpus.jsonl")]
txt = [(d["title"] + " " + d["text"]).strip() for d in random.sample(docs, 2000)]
out = {}
for name, hf in [("minilm-base", "sentence-transformers/all-MiniLM-L6-v2"),
                 ("hm-3fold", "yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold"),
                 ("ours-avg", "models/ft_avg"), ("bge-small", "BAAI/bge-small-en-v1.5")]:
    E = SentenceTransformer(hf, device="cpu").encode(txt, normalize_embeddings=True, batch_size=64)
    S = E @ E.T; iu = np.triu_indices(len(E), 1)
    out[name] = {"mean_pair_cos": float(S[iu].mean()), "std": float(S[iu].std()),
                 "effective_rank": float(np.exp(-(lambda p: (p * np.log(p)).sum())((lambda s: s / s.sum())(np.linalg.svd(E - E.mean(0), compute_uv=False)**2 + 1e-12))))}
    print(name, out[name])
json.dump(out, open("paper/results/anisotropy.json", "w"), indent=1)
