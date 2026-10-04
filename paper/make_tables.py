"""Generate paper/tex/tables.tex and fig_ndcg.pdf from paper/results/*.json (no hand-typed numbers)."""
import glob, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "paper/results"
D = {}
for f in glob.glob(f"{R}/*__*.json"):
    j = json.load(open(f)); D.setdefault(j["dataset"], {})[j["model"]] = j
DS = [("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("trec-covid", "TREC-COVID"), ("medquad", "MedQuAD")]
ROWS = [("bm25", "BM25"), ("minilm-base", "all-MiniLM-L6-v2 (base)"), ("bge-small", "BGE-small"),
        ("e5-small", "E5-small"), ("gte-small", "GTE-small"), ("medcpt", "MedCPT"),
        ("hybrid-bge-small", "BM25+BGE (RRF)"),
        ("hm-3fold", "Published HealthMate 3-fold"), ("hm-best2", "Published HealthMate best-2"),
        ("path-models_ft_avg", "Ours, seed-averaged"), ("hybrid-path-models_ft_avg", "BM25+Ours (RRF)")]
SEEDS = ["path-models_ft_s13", "path-models_ft_s42", "path-models_ft_s2024"]


def cell(ds, m):
    j = D.get(ds, {}).get(m)
    if not j: return "--"
    s = j["summary"]["ndcg@10"]; return f"{s['mean']:.3f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"


L = [r"\begin{tabular}{lcccc}", r"\toprule", "System & " + " & ".join(n for _, n in DS) + r" \\", r"\midrule"]
for m, n in ROWS:
    L.append(n + " & " + " & ".join(cell(d, m) for d, _ in DS) + r" \\")
    if m == "path-models_ft_avg":
        sd = []
        for d, _ in DS:
            v = [D[d][s]["summary"]["ndcg@10"]["mean"] for s in SEEDS if s in D.get(d, {})]
            sd.append(f"{min(v):.3f}--{max(v):.3f}" if len(v) == 3 else "--")
        L.append(r"\quad single seeds (min--max) & " + " & ".join(sd) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open("paper/tex/tab_ndcg.tex", "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(1, 4, figsize=(11, 3.4), sharey=False)
show = [("bm25", "BM25"), ("minilm-base", "MiniLM"), ("bge-small", "BGE"), ("gte-small", "GTE"), ("medcpt", "MedCPT"),
        ("hm-3fold", "HM pub."), ("path-models_ft_avg", "Ours")]
for a, (d, n) in zip(ax, DS):
    ms = [(l, D[d][m]["summary"]["ndcg@10"]) for m, l in show if m in D.get(d, {})]
    y = [s["mean"] for _, s in ms]
    err = [[s["mean"] - s["ci95"][0] for _, s in ms], [s["ci95"][1] - s["mean"] for _, s in ms]]
    a.bar(range(len(ms)), y, yerr=err, color=["#888"] * len(ms), capsize=2)
    a.set_xticks(range(len(ms))); a.set_xticklabels([l for l, _ in ms], rotation=60, ha="right", fontsize=7)
    a.set_title(n, fontsize=9); a.set_ylabel("nDCG@10" if d == "scifact" else "")
plt.tight_layout(); plt.savefig("paper/tex/fig_ndcg.pdf")
print("wrote tables and figure")

P = json.load(open(f"{R}/paired.json"))
SEL = ["hm-3fold", "path-models_ft_avg", "bge-small", "e5-small", "gte-small", "medcpt"]
NAME = dict(ROWS)
L = [r"\begin{tabular}{llccc}", r"\toprule", r"Dataset & System (vs. base) & $\Delta$ nDCG@10 & 95\% CI & Holm $p$ \\", r"\midrule"]
for ds, dn in DS:
    for r in P:
        if r["dataset"] == ds and r["b"] == "minilm-base" and r["a"] in SEL:
            L.append(f"{dn} & {NAME[r['a']]} & {r['diff']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] & {r['p_holm']:.4f} \\\\")
    for r in P:
        if r["dataset"] == ds and r["a"] == "path-models_ft_avg" and r["b"] != "minilm-base":
            L.append(f"{dn} & Ours vs. {NAME[r['b']]} & {r['diff']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] & {r['p_holm']:.4f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open("paper/tex/tab_paired.tex", "w").write("\n".join(L) + "\n")

rows = [l for l in open("paper/results/rag_summary.md").read().splitlines() if l.startswith("| base |")]
L = [r"\begin{tabular}{lcc}", r"\toprule", r"Retrieval & Accuracy [95\% CI] & $\Delta$ vs. none \\", r"\midrule"]
nm = {"none": "None", "base": "Base MiniLM", "ours": "Ours (averaged)", "hm3fold": "Published 3-fold"}
for r in sorted(rows, key=lambda l: ["none", "base", "ours", "hm3fold"].index(l.split("|")[2].strip())):
    c = [x.strip() for x in r.split("|")[1:-1]]
    L.append(f"{nm[c[1]]} & {c[2]} & {c[4] or '--'} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("paper/tex/tab_rag.tex", "w").write("\n".join(L) + "\n")
