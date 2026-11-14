"""Entropy-adaptive greedy NMS over per-frame selector scores."""

import numpy as np


def nms_interval(num_frames: int, num_select: int, scale: float) -> int:
    """Return the suppression half-width delta = round(s(u) * N / K).

    N / K is the gap uniform sampling would leave between K of N frames. It
    is floored at one frame, and so is the returned interval.
    """
    base_spacing = max(1.0, num_frames / num_select)
    return max(1, round(base_spacing * scale))


def adaptive_greedy_nms(
    scores: np.ndarray, num_select: int, interval: int
) -> np.ndarray:
    """Select frames by greedy NMS on selector scores.

    Repeatedly takes the highest-scoring frame and suppresses every frame
    within ``interval`` of it. If suppression removes all candidates before
    ``num_select`` frames are chosen, it is reset and NMS runs again over the
    frames not yet selected.

    Args:
        scores: Per-frame selector scores, shape (N,).
        num_select: Number of frames K to select.
        interval: Suppression half-width delta, in frames.

    Returns:
        Sorted indices of the selected frames, shape (min(K, N),).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a 1-D array of finite values.")
    if num_select >= scores.size:
        return np.arange(scores.size)

    available = np.ones(scores.size, dtype=bool)
    selected: list[int] = []
    while len(selected) < num_select:
        work = np.where(available, scores, -np.inf)
        while len(selected) < num_select:
            index = int(np.argmax(work))
            if not np.isfinite(work[index]):
                break
            selected.append(index)
            available[index] = False
            work[max(0, index - interval) : index + interval + 1] = -np.inf
    return np.sort(np.asarray(selected))
