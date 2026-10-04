"""Small local RAG study (PREREG amendment 3). Next-token scoring of yes/no/maybe on PubMedQA questions.
python paper/rag_eval.py --gen base|lora42|lora123|lora999 --ret none|base|ours|hm3fold"""
import argparse, json, os, time
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
LORA = "yogvidwankhede/healthmate-mistral-7b-medical-lora"
ENC = {"base": "sentence-transformers/all-MiniLM-L6-v2", "ours": "models/ft_avg",
       "hm3fold": "yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold"}
SUB = {"lora42": None, "lora123": "adapter_seed_123", "lora999": "adapter_seed_999"}
ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--ret", required=True)
ap.add_argument("--n", type=int, default=500); a = ap.parse_args()
dev = "mps"
Q = [json.loads(l) for l in open("data/pubmedqa.jsonl")][:a.n]
ctx = [""] * len(Q)
if a.ret != "none":
    chunks = json.load(open("data/rag/chunks.json")); E = np.load(f"data/rag/emb_{a.ret}.npy")
    qe = SentenceTransformer(ENC[a.ret], device=dev).encode([q["question"] for q in Q], normalize_embeddings=True)
    top = np.argsort(-(qe @ E.T), axis=1)[:, :3]
    ctx = ["\n\n".join(f"[{j + 1}] {chunks[i]['text']}" for j, i in enumerate(row)) for row in top]
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(dev)
if a.gen != "base":
    from peft import PeftModel
    kw = {"subfolder": SUB[a.gen]} if SUB[a.gen] else {}
    model = PeftModel.from_pretrained(model, LORA, **kw).to(dev)
model.eval()
cand = [tok.encode(" " + w, add_special_tokens=False)[-1] if len(tok.encode(" " + w, add_special_tokens=False)) else None for w in ["yes", "no", "maybe"]]
first = [tok.encode(w, add_special_tokens=False) for w in ["yes", "no", "maybe"]]
assert len(set(cand)) == 3, (cand, first)


def prompt(q, c):
    body = "Answer the biomedical research question with exactly one word: yes, no, or maybe."
    if c: body += "\n\nBackground text (may be irrelevant):\n" + c
    body += f"\n\nQuestion: {q}"
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True) + " Answer:"


preds, scores, t0 = [], [], time.time()
for i in range(0, len(Q), 4):
    b = tok([prompt(q["question"], c) for q, c in zip(Q[i:i + 4], ctx[i:i + 4])], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.no_grad(): lg = model(**b).logits[:, -1, :].float().cpu()
    sc = [l[cand].numpy().tolist() for l in lg]; scores += sc
    preds += [["yes", "no", "maybe"][int(np.argmax(x))] for x in sc]
    if i % 100 == 0: print(a.gen, a.ret, i, round(time.time() - t0), flush=True)
acc = float(np.mean([p == q["label"] for p, q in zip(preds, Q)]))
os.makedirs("paper/results/rag", exist_ok=True)
json.dump({"gen": a.gen, "ret": a.ret, "n": len(Q), "accuracy": acc, "seconds": round(time.time() - t0),
           "ids": [q["id"] for q in Q], "labels": [q["label"] for q in Q], "preds": preds, "scores": scores},
          open(f"paper/results/rag/{a.gen}__{a.ret}.json", "w"))
print(a.gen, a.ret, "accuracy", acc)
