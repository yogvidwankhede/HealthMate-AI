"""Zero-shot retrieval evaluation on BEIR datasets (per PREREG.md, sections 4-5).

Usage: python paper/eval_retrieval.py --dataset scifact --model bge-small
Writes paper/results/<dataset>__<model>.json with per-query metrics (so
bootstrap CIs and paired tests can be recomputed) and a summary with 95% CIs.
Models are used as published; nothing is tuned on these test sets.
"""
import argparse, csv, json, os, platform, re, time
import numpy as np

MODELS = {  # name: (hf id, query prefix, passage prefix)
    "minilm-base": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5",
                  "Represent this sentence for searching relevant passages: ", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "hm-3fold": ("yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold", "", ""),
    "hm-best2": ("yogvidwankhede/healthmate-minilm-l6-v2-medical-best2", "", ""),
    "gte-small": ("thenlper/gte-small", "", ""),
}
MEDCPT = ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder")  # public-domain licence (HF card)
SEED = 13


def load(dataset, root):
    d = os.path.join(root, dataset)
    corpus = {}
    for line in open(os.path.join(d, "corpus.jsonl")):
        r = json.loads(line)
        corpus[r["_id"]] = ((r.get("title") or "") + " " + (r.get("text") or "")).strip()
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(os.path.join(d, "queries.jsonl"))}
    qrels = {}
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {q: queries[q] for q in qrels if q in queries}
    return corpus, queries, qrels


def per_query_metrics(ranked, rel):
    """ranked: list of doc ids, best first. rel: {doc_id: gain}."""
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    mrr = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if rel.get(d, 0) > 0), 0.0)
    n = len(rel)
    return {"ndcg@10": dcg / idcg if idcg else 0.0,
            "recall@10": sum(d in rel for d in ranked[:10]) / n,
            "recall@100": sum(d in rel for d in ranked[:100]) / n,
            "mrr@10": mrr}


def bootstrap_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rank_bm25(corpus_ids, corpus_txt, qtxt):
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    bm = BM25Okapi([tok(t) for t in corpus_txt])
    return [[corpus_ids[i] for i in np.argsort(-bm.get_scores(tok(q)))[:100]] for q in qtxt]


def rank_medcpt(corpus_ids, corpus_txt, qtxt):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    def enc(hf, texts, maxlen, bs=64):
        tok, mod = AutoTokenizer.from_pretrained(hf), AutoModel.from_pretrained(hf).to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b = tok(texts[i:i + bs], truncation=True, padding=True, max_length=maxlen, return_tensors="pt").to(dev)
                out.append(mod(**b).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(out)
    Q, D = enc(MEDCPT[0], qtxt, 64), enc(MEDCPT[1], corpus_txt, 512)
    S = Q @ D.T  # MedCPT is scored by inner product
    return [[corpus_ids[i] for i in np.argsort(-s)[:100]] for s in S], 512


def rrf(lists, k=60):
    """Reciprocal rank fusion of ranked lists (Cormack et al. 2009), constant k=60."""
    sc = {}
    for lst in lists:
        for r, d in enumerate(lst):
            sc[d] = sc.get(d, 0) + 1 / (k + r + 1)
    return sorted(sc, key=sc.get, reverse=True)[:100]


def rank_dense(name, corpus_ids, corpus_txt, qtxt, max_seq=None):
    import torch
    from sentence_transformers import SentenceTransformer
    if name.startswith("path:"):  # local fine-tuned checkpoint, no prefixes
        hf, qp, pp = name[5:], "", ""
    else:
        hf, qp, pp = MODELS[name]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(hf, device=dev)
    if max_seq: m.max_seq_length = max_seq
    D = m.encode([pp + t for t in corpus_txt], batch_size=64, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    Q = m.encode([qp + q for q in qtxt], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    S = Q @ D.T
    return [[corpus_ids[i] for i in np.argsort(-s)[:100]] for s in S], m.max_seq_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="bm25 | medcpt | a key of MODELS | path:<dir> | hybrid:<model> (BM25+dense RRF)")
    ap.add_argument("--max-seq", type=int, default=None, help="override max_seq_length for dense models")
    ap.add_argument("--tag", default=None, help="name used in the output filename")
    ap.add_argument("--root", default="data/beir")
    ap.add_argument("--out", default="paper/results")
    a = ap.parse_args()
    corpus, queries, qrels = load(a.dataset, a.root)
    ids, txt = list(corpus), list(corpus.values())
    qids, qtxt = list(queries), list(queries.values())
    t0, msl = time.time(), None
    def run(m):
        if m == "bm25":
            return rank_bm25(ids, txt, qtxt), None
        if m == "medcpt":
            return rank_medcpt(ids, txt, qtxt)
        return rank_dense(m, ids, txt, qtxt, a.max_seq)
    if a.model.startswith("hybrid:"):
        (b, _), (d, msl) = run("bm25"), run(a.model[7:])
        ranked = [rrf([x, y]) for x, y in zip(b, d)]
    else:
        ranked, msl = run(a.model)
    name = a.tag or a.model.replace(":", "-").replace("/", "_")
    pq = {q: per_query_metrics(r, qrels[q]) for q, r in zip(qids, ranked)}
    summary = {}
    for k in ["ndcg@10", "recall@10", "recall@100", "mrr@10"]:
        v = [pq[q][k] for q in qids]
        summary[k] = {"mean": float(np.mean(v)), "ci95": bootstrap_ci(v)}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"dataset": a.dataset, "model": name, "spec": a.model, "n_queries": len(qids),
               "n_docs": len(ids), "max_seq_length": msl, "seconds": round(time.time() - t0, 1),
               "env": {"python": platform.python_version(), "machine": platform.machine()},
               "summary": summary, "per_query": pq},
              open(os.path.join(a.out, f"{a.dataset}__{name}.json"), "w"))
    print(a.dataset, name, {k: round(v["mean"], 4) for k, v in summary.items()})


if __name__ == "__main__":
    main()
