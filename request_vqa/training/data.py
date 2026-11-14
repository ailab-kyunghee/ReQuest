"""Pseudo-label dataset, batching and per-question segment features."""

import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset, Sampler

FEATURE_KEY = "segment_features"


def feature_path(features_dir: Union[str, Path], question_id: str) -> Path:
    """Return the file that holds the segment features of one question."""
    return Path(features_dir) / f"{question_id}.safetensors"


def align_segments(frame_segments: Sequence[int], num_frames: int) -> np.ndarray:
    """Fit per-frame segment ids to ``num_frames`` decoded frames.

    The segment ids come from 1 fps SigLIP features and may differ in length
    by a frame or two from the frames decoded now. Extra ids are dropped and
    missing ones repeat the last id.
    """
    ids = np.array(frame_segments, dtype=np.int64)[:num_frames]
    if len(ids) < num_frames:
        ids = np.concatenate([ids, np.full(num_frames - len(ids), ids[-1])])
    return ids


def segment_means(fused: torch.Tensor, frame_segments: np.ndarray) -> torch.Tensor:
    """Average fused tokens [N, T, D] over the frames of each segment.

    Segment ``c`` covers every frame whose id is ``c``, also when those frames
    are not contiguous. The result is float16 [C, T, D] with C = max id + 1; a
    segment without frames is all zeros.
    """
    ids = torch.as_tensor(frame_segments, dtype=torch.long)
    num_segments = int(ids.max()) + 1
    sums = torch.zeros(num_segments, *fused.shape[1:], dtype=torch.float32)
    sums.index_add_(0, ids, fused.float())
    counts = torch.bincount(ids, minlength=num_segments).clamp(min=1)
    return (sums / counts.view(-1, 1, 1)).half()


class PseudoLabelDataset(Dataset):
    """One item per pseudo-label: (segment-group tokens, label, question id).

    Args:
        samples: Path to ``samples.parquet`` or its DataFrame, with columns
            ``question_id``, ``segments`` (segment ids of the group) and
            ``label``. Rows of one question are consecutive.
        features_dir: Directory of per-question segment features.
        cache_size: Feature files kept in memory per worker. Consecutive
            samples share a question, so a few suffice.
    """

    def __init__(
        self,
        samples: Union[str, Path, pd.DataFrame],
        features_dir: Union[str, Path],
        cache_size: int = 8,
    ) -> None:
        table = samples
        if not isinstance(table, pd.DataFrame):
            table = pd.read_parquet(table)
        self.question_ids: List[str] = table["question_id"].tolist()
        self.segments: List[np.ndarray] = [
            np.array(s, dtype=np.int64)
            for s in table["segments"]  # writable copy
        ]
        self.labels = table["label"].to_numpy(dtype=np.float32)
        self.features_dir = Path(features_dir)
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.labels)

    def _features(self, question_id: str) -> torch.Tensor:
        if question_id in self._cache:
            self._cache.move_to_end(question_id)
            return self._cache[question_id]
        features = load_file(feature_path(self.features_dir, question_id))[FEATURE_KEY]
        self._cache[question_id] = features
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, bool, float, str]:
        """Return (tokens [T, D] float16, is_empty, label, question id).

        A group without segments gives zero tokens that are all masked, so the
        selector sees only its score token.
        """
        question_id = self.question_ids[index]
        features = self._features(question_id)
        segments = self.segments[index]
        if len(segments) == 0:
            tokens = torch.zeros(features.shape[1:], dtype=torch.float16)
            return tokens, True, float(self.labels[index]), question_id
        chosen = torch.as_tensor(segments).clamp(0, len(features) - 1)
        tokens = features[chosen].float().mean(dim=0).half()
        return tokens, False, float(self.labels[index]), question_id


def collate(
    batch: Sequence[Tuple[torch.Tensor, bool, float, str]],
) -> Dict[str, object]:
    """Pad token sequences to the longest one; padding_mask is True on padding."""
    max_len = max(tokens.shape[0] for tokens, _, _, _ in batch)
    dim = batch[0][0].shape[1]
    tokens = torch.zeros(len(batch), max_len, dim, dtype=torch.float16)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for row, (item, is_empty, _, _) in enumerate(batch):
        tokens[row, : len(item)] = item
        padding_mask[row, : len(item)] = is_empty
    return {
        "tokens": tokens,
        "padding_mask": padding_mask,
        "labels": torch.tensor([label for _, _, label, _ in batch]),
        "question_ids": [qid for _, _, _, qid in batch],
    }


class ContiguousBatchSampler(Sampler):
    """Batches of consecutive samples, in a shuffled batch order.

    Consecutive samples mostly share a question, so each batch holds whole
    groups for the ranking loss. Each process takes a contiguous share of the
    samples, and the batch order is reshuffled reproducibly with
    :meth:`set_epoch`. Every process yields the same number of full batches
    (that of the smallest share, the last one), because distributed training
    waits for all processes at every step; leftover samples are dropped.
    """

    def __init__(
        self, num_samples: int, batch_size: int, world_size: int = 1, rank: int = 0
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _per_rank(self) -> int:
        return math.ceil(self.num_samples / self.world_size)

    def __len__(self) -> int:
        smallest = self.num_samples - (self.world_size - 1) * self._per_rank()
        return max(smallest, 0) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        start = self.rank * self._per_rank()
        batches = [
            list(range(start + i * self.batch_size, start + (i + 1) * self.batch_size))
            for i in range(len(self))
        ]
        random.Random(self.epoch * 1009 + self.rank * 97).shuffle(batches)
        return iter(batches)
