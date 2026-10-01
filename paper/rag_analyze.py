"""Accuracy / macro-F1 with bootstrap CIs and paired differences vs the same generator's 'none' condition."""
import glob, json
import numpy as np
R = {}
for f in glob.glob("paper/results/rag/*__*.json"):
    j = json.load(open(f)); R[(j["gen"], j["ret"])] = j
lab = ["yes", "no", "maybe"]
def f1(y, p):
    s = []
    for c in lab:
        tp = sum(a == c and b == c for a, b in zip(y, p)); fp = sum(a != c and b == c for a, b in zip(y, p)); fn = sum(a == c and b != c for a, b in zip(y, p))
        s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(s))
rng = np.random.default_rng(13); out = ["| generator | retrieval | accuracy [95% CI] | macro-F1 | diff vs none [95% CI] | pred yes/no/maybe |", "|---|---|---|---|---|---|"]
for (g, r), j in sorted(R.items()):
    y = np.array(j["labels"]); p = np.array(j["preds"]); c = (y == p).astype(float)
    idx = rng.integers(0, len(c), (10000, len(c))); ci = np.percentile(c[idx].mean(1), [2.5, 97.5])
    d = ""
    if r != "none" and (g, "none") in R:
        c0 = (np.array(R[(g, "none")]["labels"]) == np.array(R[(g, "none")]["preds"])).astype(float)
        dd = (c - c0)[idx].mean(1); d = f"{(c - c0).mean():+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]"
    cnt = "/".join(str(int((p == l).sum())) for l in lab)
    out.append(f"| {g} | {r} | {c.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {f1(list(y), list(p)):.3f} | {d} | {cnt} |")
maj = max(lab, key=lambda l: sum(x == l for x in next(iter(R.values()))["labels"]))
out.append(f"\nMajority-class ('{maj}') accuracy: {np.mean([x == maj for x in next(iter(R.values()))['labels']]):.3f}")
open("paper/results/rag_summary.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
