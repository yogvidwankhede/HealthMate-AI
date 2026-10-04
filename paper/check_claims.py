"""Asserts that numbers quoted in prose in paper/tex/paper.tex match paper/results/*. Exit 1 on any mismatch."""
import json, glob, sys
R = "paper/results"
g = lambda d, m: json.load(open(f"{R}/{d}__{m}.json"))
nd = lambda d, m: g(d, m)["summary"]["ndcg@10"]["mean"]
P = {(r["dataset"], r["a"], r["b"]): r for r in json.load(open(f"{R}/paired.json"))}
A = json.load(open(f"{R}/anisotropy.json"))
ours = "path-models_ft_avg"
checks = [
 ("SciFact published 3-fold 0.249", round(nd("scifact", "hm-3fold"), 3) == 0.249),
 ("SciFact base 0.645", round(nd("scifact", "minilm-base"), 3) == 0.645),
 ("ours-base SciFact -0.065", round(P[("scifact", ours, "minilm-base")]["diff"], 3) == -0.065),
 ("ours-base NFCorpus -0.023", round(P[("nfcorpus", ours, "minilm-base")]["diff"], 3) == -0.023),
 ("ours-base TREC -0.029", round(P[("trec-covid", ours, "minilm-base")]["diff"], 3) == -0.029),
 ("ours-base MedQuAD -0.095", round(P[("medquad", ours, "minilm-base")]["diff"], 3) == -0.095),
 ("TREC CI -0.067..+0.007", [round(x, 3) for x in P[("trec-covid", ours, "minilm-base")]["ci"]] == [-0.067, 0.007]),
 ("TREC ours-base not sig (Holm>0.05)", P[("trec-covid", ours, "minilm-base")]["p_holm"] > 0.05),
 ("SciFact/NFC/MedQuAD sig", all(P[(d, ours, "minilm-base")]["p_holm"] < 0.05 for d in ["scifact", "nfcorpus", "medquad"])),
 ("MedCPT MedQuAD 0.535 vs 0.642", round(nd("medquad", "medcpt"), 3) == 0.535 and round(nd("medquad", "minilm-base"), 3) == 0.642),
 ("BM25 SciFact 0.652, fusion 0.656", round(nd("scifact", "bm25"), 3) == 0.652 and round(nd("scifact", "hybrid-path-models_ft_avg"), 3) == 0.656),
 ("fusion > ours on all four", all(nd(d, "hybrid-path-models_ft_avg") > nd(d, ours) for d in ["scifact", "nfcorpus", "trec-covid", "medquad"])),
 ("BGE,GTE > base on all four", all(nd(d, m) > nd(d, "minilm-base") for d in ["scifact", "nfcorpus", "trec-covid", "medquad"] for m in ["bge-small", "gte-small"])),
 ("E5 > base on TREC, MedQuAD; not sig SciFact/NFC", nd("trec-covid", "e5-small") > nd("trec-covid", "minilm-base") and P[("scifact", "e5-small", "minilm-base")]["p_holm"] > 0.05 and P[("nfcorpus", "e5-small", "minilm-base")]["p_holm"] > 0.05),
 ("MedCPT SciFact/NFC sig +, TREC ns", P[("scifact", "medcpt", "minilm-base")]["p_holm"] < 0.05 and P[("nfcorpus", "medcpt", "minilm-base")]["p_holm"] < 0.05 and P[("trec-covid", "medcpt", "minilm-base")]["p_holm"] > 0.05),
 ("ranks 138.5/27.5/137.6/162.5", [round(A[k]["effective_rank"], 1) for k in ["minilm-base", "hm-3fold", "ours-avg", "bge-small"]] == [138.5, 27.5, 137.6, 162.5]),
 ("cosines .129/.517/.123/.628", [round(A[k]["mean_pair_cos"], 3) for k in ["minilm-base", "hm-3fold", "ours-avg", "bge-small"]] == [0.129, 0.517, 0.123, 0.628]),
 ("dataset sizes", [g(d, "bm25")["n_queries"] for d in ["scifact", "nfcorpus", "trec-covid", "medquad"]] == [300, 323, 50, 15395] and [g(d, "bm25")["n_docs"] for d in ["scifact", "nfcorpus", "trec-covid", "medquad"]] == [5183, 3633, 171332, 14798]),
 ("val: base 0.997", abs(json.load(open(f"{R}/train/grid_lr1e-5_e1.json"))["base_val_mrr@10"] - 0.997) < 5e-4),
 ("neardup 4 answers > 0.8", json.load(open(f"{R}/neardup.json"))["n_answers_over_0.8"] == 4),
]
AN = json.load(open(f"{R}/adapter_nan.json"))
checks.append(("all 448 tensors NaN in all 3 adapters", all(v["tensors"] == 448 and v["tensors_with_nan"] == 448 for v in AN.values()) and len(AN) == 3))
RG = {k: json.load(open(f"{R}/rag/{k}.json")) for k in ["base__none", "base__base", "base__ours", "base__hm3fold"]}
acc = {k: v["accuracy"] for k, v in RG.items()}
checks.append(("RAG acc 0.156/0.162/0.160/0.150", [round(acc[k], 3) for k in ["base__none", "base__base", "base__ours", "base__hm3fold"]] == [0.156, 0.162, 0.16, 0.15]))
checks.append(("max |diff vs none| <= 0.006", max(abs(acc[k] - acc["base__none"]) for k in acc) <= 0.0061))
checks.append(("maybe share 88-93%", all(0.88 <= sum(p == "maybe" for p in v["preds"]) / 500 <= 0.93 for v in RG.values())))
checks.append(("majority yes 0.552", round(sum(l == "yes" for l in RG["base__none"]["labels"]) / 500, 3) == 0.552))
S5 = json.load(open(f"{R}/seq512.json"))
checks.append(("seq512 MedQuAD 0.591/0.529/0.164", [round(S5["ndcg@10"][k], 3) for k in ["base", "ours", "published"]] == [0.591, 0.529, 0.164]))
checks.append(("seq512 ours-base -0.063 and CI excludes 0", round(S5["ours_minus_base"]["diff"], 3) == -0.063 and S5["ours_minus_base"]["ci"][1] < 0))
bad = [n for n, ok in checks if not ok]
for n, ok in checks: print("OK  " if ok else "FAIL", n)
sys.exit(1 if bad else 0)
