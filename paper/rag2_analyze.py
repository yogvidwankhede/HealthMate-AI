"""PubMedQA yes/no RAG v2 (PREREG amendment 7): accuracy, retrieval hit@1, paired difference vs no context."""
import glob, json
import numpy as np
R = {}
for f in glob.glob("paper/results/rag2/*.json"):
    j = json.load(open(f)); R[j["ret"]] = j
rng = np.random.default_rng(13); none = np.array([p == l for p, l in zip(R["none"]["preds"], R["none"]["labels"])], float)
order = ["none", "oracle", "base", "ours", "hm3fold", "bge"] + sorted(k for k in R if k.startswith("path:"))
L = ["| context | hit@1 | accuracy [95% CI] | diff vs none [95% CI] |", "|---|---|---|---|"]
for k in order:
    if k not in R: continue
    c = np.array([p == l for p, l in zip(R[k]["preds"], R[k]["labels"])], float); idx = rng.integers(0, len(c), (10000, len(c)))
    ci = np.percentile(c[idx].mean(1), [2.5, 97.5]); h = R[k]["retrieval_hit@1"]
    d = "" if k == "none" else (lambda dd: f"{(c - none).mean():+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]")((c - none)[idx].mean(1))
    L.append(f"| {k} | {'' if h is None else f'{h:.3f}'} | {c.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {d} |")
L.append(f"\nn = {R['none']['n']} yes/no questions; always-yes accuracy {np.mean([l == 'yes' for l in R['none']['labels']]):.3f}")
open("paper/results/rag2_summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
