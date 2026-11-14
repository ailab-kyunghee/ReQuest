from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from request_vqa.selector import QuestionAwareSelector
from request_vqa.training.data import (
    FEATURE_KEY,
    ContiguousBatchSampler,
    PseudoLabelDataset,
    align_segments,
    collate,
    feature_path,
    segment_means,
)
from request_vqa.training.losses import group_indices, ranknet_loss, selector_loss


def test_segment_means_average_every_frame_of_a_cluster() -> None:
    # Cluster 1 appears twice: frames 1-2 and frame 4.
    fused = torch.arange(5, dtype=torch.float16).view(5, 1, 1).expand(5, 2, 3)
    means = segment_means(fused, np.array([0, 1, 1, 2, 1]))
    assert means.shape == (3, 2, 3) and means.dtype == torch.float16
    expected = torch.tensor([0.0, (1 + 2 + 4) / 3, 3.0]).half()
    assert torch.equal(means[:, 0, 0], expected)


def test_align_segments_trims_or_repeats_last_id() -> None:
    assert align_segments([0, 0, 1, 2], 3).tolist() == [0, 0, 1]
    assert align_segments([0, 1], 4).tolist() == [0, 1, 1, 1]


def _write_dataset(tmp_path: Path) -> PseudoLabelDataset:
    for qid, length in (("a", 3), ("b", 5)):
        features = torch.arange(4 * length * 2, dtype=torch.float16).view(4, length, 2)
        save_file({FEATURE_KEY: features}, str(feature_path(tmp_path, qid)))
    samples = pd.DataFrame(
        {
            "question_id": ["a", "a", "b"],
            "segments": [[0, 2], [], [3, 9]],  # 9 is out of range
            "label": [0.5, -0.1, 0.2],
        }
    )
    return PseudoLabelDataset(samples, tmp_path)


def test_dataset_averages_selected_segments(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    tokens, is_empty, label, qid = dataset[0]
    features = torch.arange(24, dtype=torch.float16).view(4, 3, 2)
    assert torch.equal(tokens, features[[0, 2]].float().mean(0).half())
    assert (is_empty, label, qid) == (False, 0.5, "a")
    tokens, _, _, _ = dataset[2]  # out-of-range ids clamp to the last segment
    features = torch.arange(40, dtype=torch.float16).view(4, 5, 2)
    assert torch.equal(tokens, features[[3, 3]].float().mean(0).half())


def test_dataset_copies_read_only_segment_arrays(tmp_path: Path) -> None:
    # Parquet gives read-only arrays, which torch warns about.
    features = torch.zeros(2, 3, 2).half()
    save_file({FEATURE_KEY: features}, str(feature_path(tmp_path, "a")))
    frame = pd.DataFrame({"question_id": ["a"], "segments": [[0, 1]], "label": [0.1]})
    frame.to_parquet(tmp_path / "samples.parquet")
    dataset = PseudoLabelDataset(tmp_path / "samples.parquet", tmp_path)
    assert all(s.flags.writeable for s in dataset.segments)


def test_collate_pads_and_masks_empty_groups(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    batch = collate([dataset[i] for i in range(3)])
    assert batch["tokens"].shape == (3, 5, 2)
    assert batch["padding_mask"].tolist() == [
        [False, False, False, True, True],
        [True] * 5,  # no segments: only the score token is visible
        [False] * 5,
    ]
    assert batch["question_ids"] == ["a", "a", "b"]


def test_padding_does_not_change_scores() -> None:
    selector = QuestionAwareSelector().eval()
    tokens = torch.randn(1, 4, 768)
    padded = torch.cat([tokens, torch.randn(1, 3, 768)], dim=1)
    mask = torch.tensor([[False] * 4 + [True] * 3])
    assert torch.allclose(selector(tokens), selector(padded, mask), atol=1e-5)


def test_sampler_keeps_consecutive_batches_and_shuffles_order() -> None:
    sampler = ContiguousBatchSampler(10, 3)
    sampler.set_epoch(0)
    batches = list(sampler)
    assert sorted(batches) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]  # last one dropped
    assert len(sampler) == 3
    # Two processes: samples 0-4 and 5-9, each in batches of 2.
    shares = [
        sorted(sum(ContiguousBatchSampler(10, 2, world_size=2, rank=r), []))
        for r in (0, 1)
    ]
    assert shares == [[0, 1, 2, 3], [5, 6, 7, 8]]


def test_sampler_gives_every_process_the_same_number_of_batches() -> None:
    # Shares of 8 and 7 samples: 2 and 1 full batches of 4, so both yield 1.
    # Otherwise distributed training would hang waiting for the shorter one.
    samplers = [ContiguousBatchSampler(15, 4, world_size=2, rank=r) for r in (0, 1)]
    assert [len(s) for s in samplers] == [1, 1]
    assert [list(s) for s in samplers] == [[[0, 1, 2, 3]], [[8, 9, 10, 11]]]
    assert len(ContiguousBatchSampler(3, 4, world_size=2, rank=0)) == 0


def test_ranknet_prefers_correct_order() -> None:
    labels = torch.tensor([1.0, 0.0])
    groups = group_indices(["q", "q"])
    good = ranknet_loss(torch.tensor([2.0, -2.0]), labels, groups, delta=0.05)
    bad = ranknet_loss(torch.tensor([-2.0, 2.0]), labels, groups, delta=0.05)
    assert good < bad


def test_ranknet_skips_groups_without_a_clear_pair() -> None:
    groups = group_indices(["q", "q", "single"])
    labels = torch.tensor([0.1, 0.12, 0.9])
    loss = ranknet_loss(torch.randn(3), labels, groups, delta=0.05)
    assert loss.item() == 0.0


def test_selector_loss_mixes_regression_and_ranking() -> None:
    scores, labels = torch.tensor([0.3, 0.1]), torch.tensor([0.5, -0.2])
    regression = torch.nn.functional.smooth_l1_loss(scores, labels, beta=0.5)
    ranking = ranknet_loss(scores, labels, group_indices(["q", "q"]), 0.05)
    expected = 0.7 * regression + 0.3 * ranking
    assert selector_loss(scores, labels, ["q", "q"]) == pytest.approx(expected.item())
