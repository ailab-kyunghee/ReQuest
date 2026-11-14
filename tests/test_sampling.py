import numpy as np
import pytest

from request_vqa.sampling import adaptive_greedy_nms, nms_interval


def _single_pass_nms(scores: np.ndarray, k: int, interval: int) -> list[int]:
    """Greedy NMS without repetition, as in the research code."""
    work = scores.astype(float).copy()
    selected = []
    for _ in range(min(k, scores.size)):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]):
            break
        selected.append(i)
        work[max(0, i - interval) : i + interval + 1] = -np.inf
    return sorted(selected)


def test_nms_interval_uses_frames_per_selected_frame() -> None:
    assert nms_interval(320, 32, 1.0) == 10
    assert nms_interval(320, 32, 0.2) == 2
    assert nms_interval(320, 8, 1.0) == 40


def test_nms_interval_is_at_least_one() -> None:
    assert nms_interval(10, 32, 0.2) == 1


def test_returns_k_sorted_unique_indices() -> None:
    scores = np.random.default_rng(0).normal(size=500)
    selected = adaptive_greedy_nms(scores, 32, interval=5)
    assert len(selected) == 32
    assert len(set(selected.tolist())) == 32
    assert np.all(np.diff(selected) > 0)


def test_matches_single_pass_nms_when_not_exhausted() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        scores = rng.normal(size=1000)
        expected = _single_pass_nms(scores, 32, interval=6)
        assert len(expected) == 32
        assert adaptive_greedy_nms(scores, 32, 6).tolist() == expected


def test_repeats_nms_when_suppression_exhausts_candidates() -> None:
    scores = np.random.default_rng(2).normal(size=320)
    first_pass = _single_pass_nms(scores, 32, interval=10)
    assert len(first_pass) < 32
    selected = adaptive_greedy_nms(scores, 32, interval=10)
    assert len(selected) == 32
    assert set(first_pass) <= set(selected.tolist())


def test_short_video_selects_every_frame() -> None:
    assert adaptive_greedy_nms(np.arange(20.0), 32, 3).tolist() == list(range(20))


def test_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError):
        adaptive_greedy_nms(np.array([0.1, np.nan, 0.3]), 2, 1)
