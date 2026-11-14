import importlib.util
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

from request_vqa.training.data import FEATURE_KEY, feature_path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_selector.py"
_spec = importlib.util.spec_from_file_location("train_selector", _SCRIPT)
train_selector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_selector)


def _make_data(root: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    rows = []
    for q in range(4):
        features = torch.randn(5, 3 + q, 768, generator=generator).half()
        save_file(
            {FEATURE_KEY: features}, str(feature_path(root / "features", f"q{q}"))
        )
        for s in range(6):
            rows.append((f"q{q}", [s % 5, (s + 1) % 5], float(s) / 5 - 0.5))
    frame = pd.DataFrame(rows, columns=["question_id", "segments", "label"])
    frame.to_parquet(root / "samples.parquet")


def _train(root: Path, run: str, epochs: int, resume: bool = False) -> None:
    args = ["--data_dir", str(root), "--features_dir", str(root / "features")]
    args += ["--output_dir", str(root / run), "--epochs", str(epochs)]
    args += ["--batch_size", "6", "--num_workers", "0", "--device", "cpu"]
    train_selector.main(args + (["--resume"] if resume else []))


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    (tmp_path / "features").mkdir()
    _make_data(tmp_path)
    _train(tmp_path, "straight", epochs=2)
    _train(tmp_path, "resumed", epochs=1)
    _train(tmp_path, "resumed", epochs=2, resume=True)
    straight = torch.load(tmp_path / "straight" / "epoch_2.pth", weights_only=True)
    resumed = torch.load(tmp_path / "resumed" / "epoch_2.pth", weights_only=True)
    assert straight.keys() == resumed.keys()
    assert all(torch.equal(straight[k], resumed[k]) for k in straight)
