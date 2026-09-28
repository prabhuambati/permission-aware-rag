from app.evaluation import (
    expected_term_coverage,
    lexical_groundedness,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.rag_types import Chunk


CHUNKS = [
    Chunk(1, "Distractor", "unrelated text", 0.9),
    Chunk(2, "Relevant policy", "Customers can request refunds within 30 days.", 0.8),
]


def test_ranking_metrics_capture_relevant_result_position():
    relevant = {"Relevant policy"}
    assert recall_at_k(CHUNKS, relevant, 1) == 0.0
    assert recall_at_k(CHUNKS, relevant, 2) == 1.0
    assert reciprocal_rank_at_k(CHUNKS, relevant, 2) == 0.5
    assert 0.0 < ndcg_at_k(CHUNKS, relevant, 2) < 1.0


def test_groundedness_and_expected_term_coverage():
    answer = "Customers can request refunds within 30 days."
    assert lexical_groundedness(answer, CHUNKS[1:]) == 1.0
    assert expected_term_coverage(answer, {"30", "days"}) == 1.0
    assert lexical_groundedness("Mars has oceans", CHUNKS[1:]) == 0.0
