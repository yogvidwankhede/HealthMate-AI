"""500 PubMedQA (pqa_labeled, MIT) questions, seed 13 -> data/pubmedqa.jsonl. Question only, no abstract."""
import json, random
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
rows = [{"id": str(r["pubid"]), "question": r["question"], "label": r["final_decision"]} for r in ds]
random.Random(13).shuffle(rows)
rows = rows[:500]
with open("data/pubmedqa.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
from collections import Counter
print(len(rows), Counter(r["label"] for r in rows))
