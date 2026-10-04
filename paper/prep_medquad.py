"""Convert MedQuAD (CC BY 4.0) to BEIR format for question -> answer retrieval.

Sources used: CancerGov, GARD, GHR, NIDDK, NINDS, SeniorHealth, NHLBI, CDC.
Excluded (PREREG amendment 1): MedlinePlus Health Topics (overlaps the training
corpus) and the three sub-collections whose answers were removed for copyright.
Corpus = de-duplicated answers; a question's relevant set = every corpus entry
with the same answer text.
"""
import glob, hashlib, json, os, re, xml.etree.ElementTree as ET
ROOT, OUT = "data/medquad", "data/beir/medquad"
SRC = ["1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA", "5_NIDDK_QA", "6_NINDS_QA", "7_SeniorHealth_QA",
       "8_NHLBI_QA_XML", "9_CDC_QA"]
clean = lambda t: re.sub(r"\s+", " ", (t or "")).strip()
corpus, queries, qrels = {}, {}, []
for s in SRC:
    for f in sorted(glob.glob(f"{ROOT}/{s}/*.xml")):
        for qa in ET.parse(f).getroot().iter("QAPair"):
            q, a = clean(qa.findtext("Question")), clean(qa.findtext("Answer"))
            if not q or len(a) < 50:
                continue
            did = hashlib.md5(a.encode()).hexdigest()[:12]
            corpus.setdefault(did, {"_id": did, "title": "", "text": a, "source": s})
            qid = s + ":" + os.path.basename(f) + ":" + qa.find("Question").get("qid")
            queries[qid] = q
            qrels.append((qid, did))
os.makedirs(f"{OUT}/qrels", exist_ok=True)
with open(f"{OUT}/corpus.jsonl", "w") as f:
    for d in corpus.values(): f.write(json.dumps(d) + "\n")
with open(f"{OUT}/queries.jsonl", "w") as f:
    for k, v in queries.items(): f.write(json.dumps({"_id": k, "text": v}) + "\n")
with open(f"{OUT}/qrels/test.tsv", "w") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for q, d in qrels: f.write(f"{q}\t{d}\t1\n")
print(len(queries), "queries;", len(corpus), "unique answers")
