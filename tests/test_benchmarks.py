import pytest

from request_vqa.benchmarks import num_options, question_with_options


def test_videomme_appends_lettered_options() -> None:
    doc = {"question": "What is shown?", "options": ["A. a cat", "B. a dog"]}
    assert (
        question_with_options(doc, "videomme") == "What is shown?\nA. a cat\nB. a dog"
    )
    assert num_options(doc, "videomme") == 2


def test_longvideobench_letters_options_and_skips_missing() -> None:
    doc = {
        "question": "Q?",
        "option0": "x",
        "option1": "y",
        "option2": "z",
        "option3": "N/A",
        "option4": "N/A",
    }
    text = question_with_options(doc, "longvideobench_val_v")
    assert text == "Q?\nA. x\nB. y\nC. z"
    assert num_options(doc, "longvideobench_val_v") == 3


def test_mlvu_uses_question_field_as_is() -> None:
    doc = {"question": "Q?\n(A) x\n(B) y\n", "candidates": ["x", "y"]}
    assert question_with_options(doc, "mlvu_dev") == doc["question"]
    assert num_options(doc, "mlvu_dev") == 2


def test_unsupported_task_raises() -> None:
    with pytest.raises(ValueError):
        num_options({}, "egoschema")
