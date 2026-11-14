"""Reading back the per-question records the model writes, for resuming runs."""

import glob
import json
import os
from typing import Dict, Tuple


def load_answers(log_dir: str) -> Dict[Tuple[str, int], str]:
    """Return final answers found in ``<log_dir>/request_rank*.jsonl``.

    Records from every rank are read, so a run can resume with a different
    number of GPUs. Records without a final answer, and a line cut off when a
    run was killed, are skipped.
    """
    answers = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "request_rank*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "answer" in record:
                    answers[(record["task"], record["doc_id"])] = record["answer"]
    return answers
