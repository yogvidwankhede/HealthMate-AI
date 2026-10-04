# Data card: fine-tuning set built for this study

- Source: MedlinePlus health topics XML, 2026-09-30 (NLM; public-domain health-topic text,
  attribution "Source: MedlinePlus, National Library of Medicine"). English topics only: 1,012.
- Excluded: A.D.A.M. encyclopedia, ASHP drug monographs, images (copyrighted per NLM terms).
- Construction: `paper/train_finetune.py`. Topic summary cut into chunks of at most 120 words;
  pairs (title, chunk), (meta description, chunk). 919 training topics, 93 validation topics
  (fixed md5 hash split of topic id), 6,666 training pairs. No LLM-generated text, no human labels.
- Not released as text by this repository; rebuild it from the NLM file with the script.
- Known limits: one source, one date; chunk/title pairs are an easy proxy task (base model
  already scores 0.997 MRR@10 on the validation split).
- Not for: training medical advice systems or evaluating clinical accuracy.
