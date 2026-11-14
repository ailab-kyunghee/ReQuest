from pathlib import Path

import torch

from request_vqa.selector import QuestionAwareSelector, load_selector, score_frames


def test_scores_one_value_per_frame() -> None:
    selector = QuestionAwareSelector().eval()
    tokens = torch.randn(5, 12, 768, dtype=torch.float16)
    scores = score_frames(selector, tokens, batch_size=2)
    assert scores.shape == (5,)


def test_batching_does_not_change_scores() -> None:
    selector = QuestionAwareSelector().eval()
    tokens = torch.randn(6, 9, 768)
    assert torch.allclose(
        torch.from_numpy(score_frames(selector, tokens, batch_size=6)),
        torch.from_numpy(score_frames(selector, tokens, batch_size=1)),
        atol=1e-5,
    )


def test_load_ignores_unused_text_branch(tmp_path: Path) -> None:
    state = QuestionAwareSelector().state_dict()
    state["t_proj.weight"] = torch.zeros(512, 1152)
    state["t_proj.bias"] = torch.zeros(512)
    state["t_pos.pe"] = torch.zeros(4096, 512)
    path = tmp_path / "selector.pth"
    torch.save(state, path)
    selector = load_selector(path, "cpu")
    assert not selector.training


def test_load_released_checkpoint_keeps_weights(tmp_path: Path) -> None:
    original = QuestionAwareSelector().eval()
    path = tmp_path / "request_selector.pth"
    torch.save(original.state_dict(), path)
    loaded = load_selector(path, "cpu")
    tokens = torch.randn(3, 7, 768)
    assert torch.equal(original(tokens), loaded(tokens))
