import torch

from request_vqa.blip_encoder import rebatch


def test_rebatch_regroups_chunks_in_order() -> None:
    frames = torch.arange(37).float()
    chunks = list(frames.split(16))  # 16, 16, 5
    batches = list(rebatch(chunks, 10))
    assert [len(b) for b in batches] == [10, 10, 10, 7]
    assert torch.equal(torch.cat(batches), frames)


def test_rebatch_keeps_batches_when_sizes_match() -> None:
    frames = torch.arange(256).float()
    batches = list(rebatch(frames.split(16), 128))
    assert [len(b) for b in batches] == [128, 128]
