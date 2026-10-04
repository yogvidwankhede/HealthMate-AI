"""Uniform weight average of SentenceTransformer checkpoints (same architecture/init)."""
import sys, torch
from sentence_transformers import SentenceTransformer
out, srcs = sys.argv[1], sys.argv[2:]
models = [SentenceTransformer(s, device="cpu") for s in srcs]
avg = {k: sum(m.state_dict()[k].float() for m in models) / len(models) for k in models[0].state_dict()}
models[0].load_state_dict(avg)
models[0].save(out)
print("averaged", len(models), "models ->", out)
