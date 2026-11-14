"""Stock lmms-eval MLVU helpers, with the option letter read from the answer."""

import re
from typing import Any, Dict, List

from lmms_eval.tasks.mlvu.utils import (  # noqa: F401
    mlvu_aggregate_results_dev,
    mlvu_doc_to_text,
    mlvu_doc_to_visual_dev,
)

# An option letter at the start of the answer, optionally after "(" and not
# followed by another letter: "C", "C.", "(C)", "C) Forest", "C. Forest".
_LEADING_OPTION = re.compile(r"\s*\(?([A-J])(?![A-Za-z])")


def extract_option_letter(answer: str) -> str:
    """Return the option letter an answer starts with, or the stripped answer."""
    match = _LEADING_OPTION.match(answer)
    return match.group(1) if match else answer.strip()


def mlvu_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Score one answer like the stock MLVU task, reading its option letter.

    The stock task takes the letter only from answers that contain ")", so
    "C." is scored as wrong even when C is correct.
    """
    return {
        "mlvu_percetion_score": {
            "question_id": doc["question"],
            "task_type": doc["task_type"],
            "pred_answer": extract_option_letter(results[0]),
            "answer": doc["answer"],
        }
    }
