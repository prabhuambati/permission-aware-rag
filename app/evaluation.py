from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from .rag_types import Chunk

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "for", "can", "what", "who", "within", "i", "could", "not", "find", "answer"}


@dataclass
class EvaluationCase:
    case_id: str
    user: str
    question: str
    relevant_titles: set[str]
    answer_terms: set[str]
    answerable: bool = True


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS}


def recall_at_k(chunks: list[Chunk], relevant_titles: set[str], k: int) -> float:
    if not relevant_titles:
        return 1.0 if not chunks else 0.0
    retrieved = {chunk.title for chunk in chunks[:k]}
    return 1.0 if retrieved & relevant_titles else 0.0


def reciprocal_rank_at_k(chunks: list[Chunk], relevant_titles: set[str], k: int) -> float:
    if not relevant_titles:
        return 1.0 if not chunks else 0.0
    for index, chunk in enumerate(chunks[:k], start=1):
        if chunk.title in relevant_titles:
            return 1.0 / index
    return 0.0


def ndcg_at_k(chunks: list[Chunk], relevant_titles: set[str], k: int) -> float:
    if not relevant_titles:
        return 1.0 if not chunks else 0.0
    relevance = [1 if chunk.title in relevant_titles else 0 for chunk in chunks[:k]]
    dcg = sum(score / math.log2(index + 2) for index, score in enumerate(relevance))
    ideal_count = min(len(relevant_titles), k)
    ideal = sum(1 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def lexical_groundedness(answer: str, chunks: list[Chunk]) -> float:
    """Proxy metric: share of meaningful answer tokens supported by retrieved context.

    This is intentionally not presented as a proof of factuality. It is a cheap regression
    signal that catches answers drifting away from the retrieved evidence.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 1.0 if not chunks else 0.0
    context_tokens = _tokens(" ".join(chunk.text for chunk in chunks))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def expected_term_coverage(answer: str, expected_terms: set[str]) -> float:
    if not expected_terms:
        return 1.0
    answer_tokens = _tokens(answer)
    return len({term.lower() for term in expected_terms} & answer_tokens) / len(expected_terms)


def evaluate_cases(
    cases: Iterable[EvaluationCase],
    retrieve: Callable[[str, str, int], list[Chunk]],
    answer: Callable[[str, str, list[Chunk]], str],
    k: int = 3,
) -> dict:
    rows = []
    for case in cases:
        chunks = retrieve(case.user, case.question, k)
        generated = answer(case.user, case.question, chunks)
        rows.append({
            "id": case.case_id,
            "answerable": case.answerable,
            "recall_at_k": recall_at_k(chunks, case.relevant_titles, k),
            "mrr_at_k": reciprocal_rank_at_k(chunks, case.relevant_titles, k),
            "ndcg_at_k": ndcg_at_k(chunks, case.relevant_titles, k),
            "groundedness": lexical_groundedness(generated, chunks),
            "answer_term_coverage": expected_term_coverage(generated, case.answer_terms),
            "access_control_pass": 1.0 if case.answerable or not chunks else 0.0,
            "retrieved_titles": [chunk.title for chunk in chunks],
            "answer": generated,
        })
    if not rows:
        return {"count": 0, "metrics": {}, "cases": []}
    answerable_rows = [row for row in rows if row["answerable"]]
    metric_names = ["recall_at_k", "mrr_at_k", "ndcg_at_k", "groundedness", "answer_term_coverage"]
    metrics = {
        name: round(sum(row[name] for row in answerable_rows) / len(answerable_rows), 4)
        for name in metric_names
    } if answerable_rows else {}
    metrics["access_control_pass_rate"] = round(sum(row["access_control_pass"] for row in rows) / len(rows), 4)
    return {
        "count": len(rows),
        "answerable_count": len(answerable_rows),
        "k": k,
        "metrics": metrics,
        "cases": rows,
    }
