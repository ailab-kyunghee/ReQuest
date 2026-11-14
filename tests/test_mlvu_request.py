import pytest

from request_vqa.tasks.mlvu_request.utils import (
    extract_option_letter,
    mlvu_process_results,
)


@pytest.mark.parametrize(
    "answer, letter",
    [
        ("C", "C"),
        ("C.", "C"),
        ("C)", "C"),
        ("(C)", "C"),
        (" B) Minecraft gameplay video", "B"),
        ("D. Communicating", "D"),
        ("Forest", "Forest"),  # no leading option letter: left for scoring as is
        ("", ""),
    ],
)
def test_extract_option_letter(answer: str, letter: str) -> None:
    assert extract_option_letter(answer) == letter


def test_process_results_scores_letter_with_period() -> None:
    doc = {"question": "q", "task_type": "needle", "answer": "C"}
    record = mlvu_process_results(doc, ["C."])["mlvu_percetion_score"]
    assert record["pred_answer"] == record["answer"] == "C"
