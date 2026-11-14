"""Uncertainty signals for re-thinking routing and NMS spacing.

Entropies are normalized by log(M), so they lie in [0, 1] for any number of
answer options M.
"""

import math
from typing import Sequence

import numpy as np


def normalized_entropy(option_probs: Sequence[float]) -> float:
    """Return the answer-option entropy divided by log(M).

    Args:
        option_probs: Probabilities the MLLM assigns to each option letter.
            They are renormalized over the M options first.

    Returns:
        Normalized entropy u in [0, 1].
    """
    probs = np.clip(np.nan_to_num(np.asarray(option_probs, dtype=np.float64)), 0, None)
    if probs.size < 2:
        raise ValueError("At least two answer options are required.")
    total = probs.sum()
    if total <= 0:
        return 1.0
    probs = probs / total
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
    return float(entropy / math.log(probs.size))


def routing_threshold(
    num_frames: int, tau0: float, gamma_len: float, max_frames: int
) -> float:
    """Return the length-aware threshold tau_eff = tau0 - gamma * min(1, N/N_max).

    Args:
        num_frames: Number of video frames N at the dense (1 fps) rate.
        tau0: Base threshold.
        gamma_len: Weight of the length term.
        max_frames: Length N_max at which the length term saturates.
    """
    return tau0 - gamma_len * min(1.0, num_frames / max_frames)


def spacing_scale(
    entropy: float, s_min: float, rho: float, u_min: float, u_max: float
) -> float:
    """Map entropy to the NMS spacing coefficient s(u) in [s_min, 1].

    Entropy below ``u_min`` keeps uniform-like spacing (s = 1). Higher entropy
    shrinks the spacing toward ``s_min``, so frames can cluster around the
    evidence.
    """
    if entropy < u_min:
        return 1.0
    clipped = min(max(entropy, u_min), u_max)
    weight = ((clipped - u_min) / (u_max - u_min)) ** rho
    return s_min + (1.0 - weight) * (1.0 - s_min)
