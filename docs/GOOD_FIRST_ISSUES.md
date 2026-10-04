# Draft "good first issue" tickets (not yet created on GitHub; create only if you want them)

1. **Pin or document a supported Python range.** `requirements.txt` fails to resolve on Python 3.14
   (`langchain-pinecone==0.2.8` unavailable). Test on 3.10-3.12 and state the range in the README.
2. **Add a unit test for the lexical metrics in `eval_metrices.py`.** The function named BLEU is
   unigram precision; add tests, and rename or document it.
3. **Make the Flask app fail clearly when API keys are missing** instead of a stack trace.
4. **Add a data card for the question set** once the author supplies provenance for the reference answers.
5. **Document the Hugging Face model-loading snippet with a pinned `transformers`/`peft` version** and verify it runs.
