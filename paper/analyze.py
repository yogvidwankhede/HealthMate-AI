"""Aggregate per-query results into tables (paper/results/summary.md) and a figure.

Paired bootstrap (10,000 resamples over queries, seed 13) of nDCG@10 differences
vs. base MiniLM, and our seed-averaged model vs. the best external baseline per dataset (chosen post hoc, so exploratory); Holm correction over
all comparisons reported. Two-sided p = 2*min(P(diff<=0), P(diff>=0)).
"""
import glob, json, os, sys
import numpy as np
R = "paper/results"
data = {}
for f in glob.glob(f"{R}/*__*.json"):
    j = json.load(open(f)); data.setdefault(j["dataset"], {})[j["model"]] = j


def paired(a, b, n=10000, seed=13):
    qs = sorted(set(a["per_query"]) & set(b["per_query"]))
    d = np.array([a["per_query"][q]["ndcg@10"] - b["per_query"][q]["ndcg@10"] for q in qs])
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()) + 1 / n)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], p


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


lines, comps = [], []
for ds, models in sorted(data.items()):
    lines += [f"\n## {ds}  (n_queries={next(iter(models.values()))['n_queries']})\n",
              "| model | nDCG@10 [95% CI] | Recall@10 | Recall@100 | MRR@10 |", "|---|---|---|---|---|"]
    for m, j in sorted(models.items(), key=lambda kv: -kv[1]["summary"]["ndcg@10"]["mean"]):
        s = j["summary"]; c = s["ndcg@10"]["ci95"]
        lines.append(f"| {m} | {s['ndcg@10']['mean']:.3f} [{c[0]:.3f}, {c[1]:.3f}] | {s['recall@10']['mean']:.3f} | "
                     f"{s['recall@100']['mean']:.3f} | {s['mrr@10']['mean']:.3f} |")
    ref, ours = "minilm-base", "path-models_ft_avg"
    if ref in models:
        for m in models:
            if m != ref: comps.append((ds, m, ref) + paired(models[m], models[ref]))
        base = [m for m in models if m in ("bge-small", "e5-small", "gte-small", "medcpt", "bm25")]
        if ours in models and base:  # best external baseline is chosen post hoc, per dataset
            best = max(base, key=lambda m: models[m]["summary"]["ndcg@10"]["mean"])
            comps.append((ds, ours, best) + paired(models[ours], models[best]))
if comps:
    adj = holm([c[5] for c in comps])
    lines += ["\n## Paired comparisons (nDCG@10 difference, Holm-adjusted over all rows)\n",
              "| dataset | A | B | mean(A-B) | 95% CI | p (raw) | p (Holm) |", "|---|---|---|---|---|---|---|"]
    seen = set()
    for c, a in zip(comps, adj):
        k = (c[0], c[1], c[2])
        if k in seen: continue
        seen.add(k)
        lines.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]:+.3f} | [{c[4][0]:+.3f}, {c[4][1]:+.3f}] | {c[5]:.4f} | {a:.4f} |")
json.dump([{"dataset": c[0], "a": c[1], "b": c[2], "diff": c[3], "ci": c[4], "p_raw": c[5], "p_holm": float(a)} for c, a in zip(comps, adj)] if comps else [], open(f"{R}/paired.json", "w"), indent=1)
open(f"{R}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
