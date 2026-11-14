"""Training loss: SmoothL1 regression plus RankNet ranking within each question."""

from typing import Dict, Hashable, Sequence

import torch
import torch.nn.functional as F


def group_indices(keys: Sequence[Hashable]) -> Dict[Hashable, torch.Tensor]:
    """Map each key with at least two items in ``keys`` to their positions."""
    groups: Dict[Hashable, list] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    return {k: torch.tensor(v) for k, v in groups.items() if len(v) >= 2}


def ranknet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    groups: Dict[Hashable, torch.Tensor],
    delta: float,
) -> torch.Tensor:
    """Pairwise logistic ranking loss within each group, averaged over groups.

    A group counts only if some pair of its labels differs by more than
    ``delta``. The loss of a counted group averages over all its pairs, as in
    the training run of the released checkpoints.
    """
    total = scores.new_zeros(())
    count = 0
    for positions in groups.values():
        positions = positions.to(scores.device)
        s, y = scores[positions], labels[positions]
        i, j = torch.triu_indices(len(positions), len(positions), 1, device=s.device)
        if ((y[i] - y[j]).abs() > delta).any():
            margin = (s[i] - s[j]) * torch.sign(y[i] - y[j])
            total = total + F.softplus(-margin).mean()
            count += 1
    return total / max(count, 1)


def selector_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    question_ids: Sequence[Hashable],
    rank_weight: float = 0.3,
    smoothl1_beta: float = 0.5,
    rank_delta: float = 0.05,
) -> torch.Tensor:
    """Return (1 - w) * SmoothL1(scores, labels) + w * RankNet within questions."""
    regression = F.smooth_l1_loss(scores, labels, beta=smoothl1_beta)
    ranking = ranknet_loss(scores, labels, group_indices(question_ids), rank_delta)
    return (1.0 - rank_weight) * regression + rank_weight * ranking
