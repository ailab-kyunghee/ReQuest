import json
from pathlib import Path

from request_vqa.answer_log import load_answers


def test_load_answers_reads_all_ranks_and_skips_incomplete(tmp_path: Path) -> None:
    rank0 = [
        {"task": "mlvu_dev", "doc_id": 0, "rethink": False, "answer": "A"},
        {"task": "mlvu_dev", "doc_id": 2, "rethink": True},  # no final answer
    ]
    (tmp_path / "request_rank0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rank0)
    )
    (tmp_path / "request_rank1.jsonl").write_text(
        json.dumps({"task": "mlvu_dev", "doc_id": 1, "answer": "C"})
        + "\n"
        + '{"task": "mlvu_dev", "doc_id": 3, "ans'  # cut off when killed
    )
    assert load_answers(str(tmp_path)) == {("mlvu_dev", 0): "A", ("mlvu_dev", 1): "C"}


def test_load_answers_on_empty_dir(tmp_path: Path) -> None:
    assert load_answers(str(tmp_path)) == {}
