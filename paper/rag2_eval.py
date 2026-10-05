"""RAG v2 (PREREG amendment 7): PubMedQA yes/no with a corpus that contains the gold abstract.
python paper/rag2_eval.py --ret none|oracle|base|hm3fold|ours|bge|<path:dir>"""
import argparse, json, os, time
import numpy as np, torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--ret", required=True); a = ap.parse_args()
ENC = {"base": ("sentence-transformers/all-MiniLM-L6-v2", "", ""), "ours": ("models/ft_avg", "", ""),
       "hm3fold": ("yogvidwankhede/healthmate-minilm-l6-v2-medical-3fold", "", ""),
       "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", "")}
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
byid = {str(r["pubid"]): r for r in ds}
Q = [q for q in map(json.loads, open("data/pubmedqa.jsonl")) if q["label"] in ("yes", "no")]
docs_id = list(byid); docs = [" ".join(byid[i]["context"]["contexts"]) for i in docs_id]
ctx, hit = [""] * len(Q), None
if a.ret == "oracle":
    ctx = [" ".join(byid[q["id"]]["context"]["contexts"]) for q in Q]
elif a.ret != "none":
    hf, qp, pp = (a.ret[5:], "", "") if a.ret.startswith("path:") else ENC[a.ret]
    m = SentenceTransformer(hf, device="mps")
    D = m.encode([pp + d for d in docs], normalize_embeddings=True, batch_size=32)
    top1 = np.argmax(m.encode([qp + q["question"] for q in Q], normalize_embeddings=True) @ D.T, axis=1)
    ctx = [docs[j] for j in top1]; hit = float(np.mean([docs_id[j] == q["id"] for j, q in zip(top1, Q)]))
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("mps").eval()
cand = [tok.encode(" " + w, add_special_tokens=False)[-1] for w in ["yes", "no"]]


def prompt(q, c):
    body = "Answer the biomedical research question with exactly one word: yes or no."
    if c: body += "\n\nContext (may be irrelevant):\n" + c
    body += f"\n\nQuestion: {q}"
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True) + " Answer:"


preds, scores, t0 = [], [], time.time()
for i in range(0, len(Q), 4):
    b = tok([prompt(q["question"], c) for q, c in zip(Q[i:i + 4], ctx[i:i + 4])], return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
    with torch.no_grad(): lg = model(**b).logits[:, -1, :].float().cpu()
    sc = [l[cand].numpy().tolist() for l in lg]; scores += sc; preds += ["yes" if x[0] >= x[1] else "no" for x in sc]
acc = float(np.mean([p == q["label"] for p, q in zip(preds, Q)]))
os.makedirs("paper/results/rag2", exist_ok=True)
name = a.ret.replace(":", "-").replace("/", "_")
json.dump({"ret": a.ret, "n": len(Q), "accuracy": acc, "retrieval_hit@1": hit, "seconds": round(time.time() - t0),
           "ids": [q["id"] for q in Q], "labels": [q["label"] for q in Q], "preds": preds, "scores": scores},
          open(f"paper/results/rag2/{name}.json", "w"))
print(a.ret, "acc", round(acc, 3), "hit@1", hit)
