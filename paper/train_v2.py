"""Stronger re-training recipes R1 (hard negatives) and R2 (+ general replay). PREREG amendment 6.
python paper/train_v2.py --recipe r1|r2 --lr 1e-5 --epochs 2 --seed 13 --out models/v2_r1_lr1e-5_e2"""
import argparse, json, os, random, sys
import numpy as np, torch
from datasets import Dataset, DatasetDict, load_dataset
sys.path.insert(0, "paper")
from train_finetune import load_topics, is_val, BASE
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True, choices=["r1", "r2"]); ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--epochs", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
T = [t for t in load_topics() if not is_val(t["id"])]
chunks = [(i, c) for i, t in enumerate(T) for c in t["chunks"]]
base = SentenceTransformer(BASE, device=dev)
C = base.encode([c for _, c in chunks], normalize_embeddings=True, batch_size=64)
owner = np.array([i for i, _ in chunks]); rng = random.Random(a.seed)
rows = []
for ti, t in enumerate(T):
    for q in (t["title"], t["meta"]):
        if not q: continue
        qe = base.encode([q], normalize_embeddings=True)[0]
        order = [j for j in np.argsort(-(C @ qe)) if owner[j] != ti]
        for j, (oi, c) in enumerate(chunks):
            if oi == ti:
                rows.append({"anchor": q, "positive": c, "negative": chunks[order[rng.randint(4, 29)]][1]})
med = Dataset.from_list(rows)
data = {"med": med}; lossd = None
model = SentenceTransformer(BASE, device=dev)
if a.recipe == "r2":
    nq = load_dataset("sentence-transformers/natural-questions", "pair", split="train").shuffle(seed=13).select(range(len(rows)))
    data["gen"] = nq.rename_columns({"query": "anchor", "answer": "positive"})
    mnrl = losses.MultipleNegativesRankingLoss(model); lossd = {"med": mnrl, "gen": mnrl}
args = SentenceTransformerTrainingArguments(
    output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, learning_rate=a.lr, seed=a.seed, per_device_train_batch_size=32,
    warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES, multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    save_strategy="no", report_to="none", logging_steps=100)
SentenceTransformerTrainer(model=model, args=args, train_dataset=DatasetDict(data) if lossd else med,
                           loss=lossd if lossd else losses.MultipleNegativesRankingLoss(model)).train()
model.save(a.out)
os.makedirs("paper/results/train_v2", exist_ok=True)
json.dump({"recipe": a.recipe, "lr": a.lr, "epochs": a.epochs, "seed": a.seed, "n_med_triplets": len(rows)},
          open(f"paper/results/train_v2/{os.path.basename(a.out)}.json", "w"), indent=1)
print("saved", a.out, len(rows))
