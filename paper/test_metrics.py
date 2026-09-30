import math
from eval_retrieval import per_query_metrics
def test_perfect():
    m = per_query_metrics(["a", "b"], {"a": 1, "b": 1})
    assert m["ndcg@10"] == 1 and m["mrr@10"] == 1 and m["recall@10"] == 1
def test_second_rank():
    m = per_query_metrics(["x", "a"], {"a": 1})
    assert abs(m["ndcg@10"] - 1 / math.log2(3)) < 1e-9 and m["mrr@10"] == 0.5
def test_miss():
    assert per_query_metrics(["x"], {"a": 1})["ndcg@10"] == 0
