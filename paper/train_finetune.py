"""Fine-tune all-MiniLM-L6-v2 on MedlinePlus (PREREG amendment 1).

python paper/train_finetune.py --lr 2e-5 --epochs 2 --seed 13 --out models/ft_s13
Prints and saves validation MRR@10 (title / meta-description -> own chunks, held-out topics).
"""
import argparse, hashlib, json, re, xml.etree.ElementTree as ET, os, random
import numpy as np, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers

XML = "data/mplus/mplus_topics_2026-09-30.xml"
BASE = "sentence-transformers/all-MiniLM-L6-v2"
strip = lambda h: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h or "")).strip()


def chunks(text, maxw=120):
    out, cur = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(" ".join(cur + [sent]).split()) > maxw:
            out.append(" ".join(cur)); cur = []
        cur.append(sent)
    if cur: out.append(" ".join(cur))
    return [c for c in out if len(c.split()) >= 8]


def load_topics():
    T = []
    for x in ET.parse(XML).getroot():
        if x.get("language") != "English": continue
        body = strip(x.findtext("full-summary"))
        ch = chunks(body)
        if ch: T.append({"id": x.get("id"), "title": x.get("title"), "meta": x.get("meta-desc") or "", "chunks": ch})
    return T


def is_val(tid):  # fixed hash split, independent of seed
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 10 == 0


def val_mrr(model, val):
    docs, owner = [], []
    for i, t in enumerate(val):
        for c in t["chunks"]: docs.append(c); owner.append(i)
    D = model.encode(docs, normalize_embeddings=True, batch_size=64)
    rr = []
    for i, t in enumerate(val):
        for q in [t["title"], t["meta"]]:
            if not q: continue
            s = D @ model.encode([q], normalize_embeddings=True)[0]
            order = np.argsort(-s)[:10]
            rr.append(next((1 / (r + 1) for r, j in enumerate(order) if owner[j] == i), 0.0))
    return float(np.mean(rr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True); ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    T = load_topics(); tr = [t for t in T if not is_val(t["id"])]; va = [t for t in T if is_val(t["id"])]
    rows = [{"anchor": q, "positive": c} for t in tr for c in t["chunks"] for q in (t["title"], t["meta"]) if q]
    model = SentenceTransformer(BASE, device="mps" if torch.backends.mps.is_available() else "cpu")
    args = SentenceTransformerTrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr, seed=a.seed,
        per_device_train_batch_size=32, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", report_to="none", logging_steps=50)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(a.out)
    res = {"lr": a.lr, "epochs": a.epochs, "seed": a.seed, "n_train_topics": len(tr), "n_val_topics": len(va),
           "n_pairs": len(rows), "val_mrr@10": val_mrr(model, va),
           "base_val_mrr@10": val_mrr(SentenceTransformer(BASE, device="cpu"), va) if a.epochs == 1 and a.lr == 1e-5 else None}
    os.makedirs("paper/results/train", exist_ok=True)
    json.dump(res, open(f"paper/results/train/{os.path.basename(a.out)}.json", "w"), indent=1)
    print(res)
