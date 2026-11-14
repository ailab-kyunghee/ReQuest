"""Benchmark-specific question text and option counts."""

from typing import Any, Mapping

OPTION_LETTERS = "ABCDEFGHIJ"
_FAMILIES = ("videomme", "longvideobench", "mlvu")


def _family(task: str) -> str:
    for family in _FAMILIES:
        if task.startswith(family):
            return family
    raise ValueError(f"ReQuest does not support task '{task}'.")


def _longvideobench_options(doc: Mapping[str, Any]) -> list[str]:
    options = (str(doc.get(f"option{i}", "N/A")) for i in range(5))
    return [option for option in options if option != "N/A"]


def question_with_options(doc: Mapping[str, Any], task: str) -> str:
    """Return the question text BLIP fuses with each frame.

    This is the text the selector was trained and evaluated with, not the
    MLLM prompt.
    """
    family = _family(task)
    question = str(doc["question"])
    if family == "videomme":
        # Video-MME options already start with "A. ", "B. ", ...
        return question + "".join(f"\n{option}" for option in doc["options"])
    if family == "longvideobench":
        options = _longvideobench_options(doc)
        lines = (f"\n{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        return question + "".join(lines)
    # MLVU questions already list their options as "(A) ...".
    return question


def num_options(doc: Mapping[str, Any], task: str) -> int:
    """Return the number of answer options M of a multiple-choice question."""
    family = _family(task)
    if family == "videomme":
        return len(doc["options"])
    if family == "longvideobench":
        return len(_longvideobench_options(doc))
    return len(doc["candidates"])
