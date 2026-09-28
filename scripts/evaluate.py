from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import db  # noqa: E402
from app.evaluation import EvaluationCase, evaluate_cases  # noqa: E402
from app.rag import authenticate, retrieve  # noqa: E402
from app.llm import get_generator  # noqa: E402


def load_cases() -> list[EvaluationCase]:
    path = PROJECT_ROOT / "evaluation" / "golden_set.json"
    raw_cases = json.loads(path.read_text())
    return [
        EvaluationCase(
            case_id=item["id"],
            user=item["user"],
            question=item["question"],
            relevant_titles=set(item["relevant_titles"]),
            answer_terms=set(item["answer_terms"]),
            answerable=item.get("answerable", True),
        )
        for item in raw_cases
    ]


def main() -> None:
    db.init_db()
    generator = get_generator()
    cases = load_cases()

    def retrieve_for_case(username: str, question: str, k: int):
        return retrieve(authenticate(username), question, k)

    def answer_for_case(username: str, question: str, chunks):
        return generator.generate(question, chunks)

    result = evaluate_cases(cases, retrieve_for_case, answer_for_case, k=3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
