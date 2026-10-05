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
V2 = json.load(open(f"{R}/v2_summary.json")); R2 = {k: json.load(open(f"{R}/rag2/{k}.json")) for k in ["none", "oracle", "base", "ours", "hm3fold", "bge", "path-models_v2final_avg"]}
r2 = lambda k: round(R2[k]["accuracy"], 3)
d2 = lambda d, k: round(V2[d][k]["diff"], 3)
checks += [
 ("v2 minus base: -0.006/-0.005/-0.017/-0.023", [d2(d, "recipe v2, averaged minus base") for d in ["scifact", "nfcorpus", "trec-covid", "medquad"]] == [-0.006, -0.005, -0.017, -0.023]),
 ("v2 CIs include 0 on three BEIR sets; MedQuAD test Holm<0.001", all(V2[d]["recipe v2, averaged minus base"]["ci"][0] < 0 < V2[d]["recipe v2, averaged minus base"]["ci"][1] for d in ["scifact", "nfcorpus", "trec-covid"]) and V2["medquad"]["v2_holm_p"] < 0.001),
 ("first recipe MedQuAD test -0.105", d2("medquad", "recipe 1 (earlier), averaged minus base") == -0.105),
 ("TREC v2 0.455 vs BGE 0.756", round(V2["trec-covid"]["ndcg@10"]["recipe v2, averaged"][0], 3) == 0.455 and round(V2["trec-covid"]["ndcg@10"]["BGE-small"][0], 3) == 0.756),
 ("MedQuAD test n=4599", V2["medquad"]["n_queries"] == 4599),
 ("RAG2 accuracies none .631 oracle .803 base .799 ours .792 hm3 .738 bge .803 v2 .799", [r2(k) for k in ["none", "oracle", "base", "ours", "hm3fold", "bge", "path-models_v2final_avg"]] == [0.631, 0.803, 0.799, 0.792, 0.738, 0.803, 0.799]),
 ("RAG2 hit@1 .604 .930 .966 .975 .991", [round(R2[k]["retrieval_hit@1"], 3) for k in ["hm3fold", "ours", "path-models_v2final_avg", "base", "bge"]] == [0.604, 0.93, 0.966, 0.975, 0.991]),
 ("RAG2 n=442", R2["none"]["n"] == 442),
 ("RAG2 hm3fold-base -0.061 [-0.090,-0.032]", (lambda x: round(x["diff"], 3) == -0.061 and [round(c, 3) for c in x["ci"]] == [-0.09, -0.032])(json.load(open(f"{R}/rag2_paired.json"))["hm3fold_minus_base_retriever"])),
 ("dev grid: base 0.693, range 0.607-0.659", (lambda g: round(g["base"], 3) == 0.693 and round(min(g["grid"]), 3) == 0.607 and round(max(g["grid"]), 3) == 0.659)({"base": json.load(open(f"{R}/medquad__dev__minilm-base.json"))["summary"]["ndcg@10"]["mean"], "grid": [json.load(open(f))["summary"]["ndcg@10"]["mean"] for f in glob.glob(f"{R}/medquad__dev__v2_*.json")]})),
 ("dev n queries 10,796", json.load(open(f"{R}/medquad__dev__minilm-base.json"))["n_queries"] == 10796),
]
AH = json.load(open(f"{R}/adapter_hashes.json"))["files"]
checks.append(("3 adapter files share one SHA-256", len({v["sha256"] for v in AH.values()}) == 1 and len(AH) == 3))
bad = [n for n, ok in checks if not ok]
for n, ok in checks: print("OK  " if ok else "FAIL", n)
sys.exit(1 if bad else 0)
