"""Follow-up analysis (PREREG amendment 6): MedQuAD DEV/TEST split + BEIR sets for base, earlier recipe, v2 recipe.
Writes paper/results/v2_summary.json and v2_summary.md. Holm over the 4 test comparisons (v2 average vs base)."""
import json
import numpy as np
R = "paper/results"
TEST = ("1_CancerGov_QA:", "5_NIDDK_QA:", "6_NINDS_QA:", "7_SeniorHealth_QA:", "8_NHLBI_QA_XML:", "9_CDC_QA:")
SYS = {"base": "minilm-base", "published 3-fold": "hm-3fold", "recipe 1 (earlier), averaged": "path-models_ft_avg",
       "recipe v2, averaged": "v2final_avg", "BGE-small": "bge-small"}
def load(d, m):
    j = json.load(open(f"{R}/{d}__{m}.json")); return j["per_query"]
def sub(pq, d): return {q: v for q, v in pq.items() if d != "medquad" or q.startswith(TEST)}
def boot(x, n=10000, seed=13):
    x = np.asarray(x); rng = np.random.default_rng(seed); m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def paired(a, b, n=10000, seed=13):
    qs = sorted(set(a) & set(b)); d = np.array([a[q]["ndcg@10"] - b[q]["ndcg@10"] for q in qs]); rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()) + 1 / n)
out, rows, ps = {}, [], []
for d in ["scifact", "nfcorpus", "trec-covid", "medquad"]:
    P = {k: sub(load(d, m), d) for k, m in SYS.items()}
    out[d] = {"n_queries": len(P["base"]), "ndcg@10": {k: boot([v["ndcg@10"] for v in p.values()]) for k, p in P.items()}}
    for k in ["recipe v2, averaged", "recipe 1 (earlier), averaged"]:
        df = paired(P[k], P["base"]); out[d][f"{k} minus base"] = {"diff": df[0], "ci": df[1], "p_raw": df[2]}
        if k == "recipe v2, averaged": ps.append((d, df[2]))
adj = {}
order = np.argsort([p for _, p in ps]); run = 0
for rank, i in enumerate(order):
    run = max(run, (len(ps) - rank) * ps[i][1]); adj[ps[i][0]] = min(1.0, run)
for d in out: out[d]["v2_holm_p"] = adj[d]
json.dump(out, open(f"{R}/v2_summary.json", "w"), indent=1)
L = ["| dataset (n queries) | system | nDCG@10 [95% CI] |", "|---|---|---|"]
for d, o in out.items():
    for k, (m, ci) in o["ndcg@10"].items(): L.append(f"| {d} ({o['n_queries']}) | {k} | {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
L += ["", "| dataset | v2 avg minus base [95% CI] | Holm p | earlier recipe minus base [95% CI] |", "|---|---|---|---|"]
for d, o in out.items():
    a, b = o["recipe v2, averaged minus base"], o["recipe 1 (earlier), averaged minus base"]
    L.append(f"| {d} | {a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {o['v2_holm_p']:.4f} | {b['diff']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] |")
open(f"{R}/v2_summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
