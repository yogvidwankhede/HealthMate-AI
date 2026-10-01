"""Embed all MedlinePlus English chunks (120 words) with three encoders; save data/rag/*."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "paper")
from train_finetune import load_topics
from sentence_transformers import SentenceTransformer
os.makedirs("data/rag", exist_ok=True)
T = load_topics()
chunks = [{"topic": t["title"], "text": c} for t in T for c in t["chunks"]]
json.dump(chunks, open("data/rag/chunks.json", "w"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
for name, hf in [("base", "sentence-transformers/all-MiniLM-L6-v2"), ("ours", "models/ft_avg"),
                 ("hm3fold", "yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold")]:
    E = SentenceTransformer(hf, device=dev).encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=64)
    np.save(f"data/rag/emb_{name}.npy", E); print(name, E.shape)
