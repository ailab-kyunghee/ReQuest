import time
from typing import List

import numpy as np
import pytest

from request_vqa import video
from request_vqa.video import (
    dense_frame_indices,
    iter_frame_batches,
    uniform_frame_indices,
)


def test_dense_indices_step_by_rounded_fps() -> None:
    assert dense_frame_indices(90, 29.97, 1.0).tolist() == [0, 30, 60]
    assert dense_frame_indices(5, 0.4, 1.0).tolist() == [0, 1, 2, 3, 4]


def test_uniform_indices_span_whole_video_for_long_videos() -> None:
    dense = dense_frame_indices(3000, 30.0, 1.0)  # 100 dense frames
    indices = uniform_frame_indices(3000, dense, 32)
    assert len(indices) == 32
    assert indices[0] == 0 and indices[-1] == 2999


def test_uniform_indices_use_dense_frames_for_short_videos() -> None:
    dense = dense_frame_indices(600, 30.0, 1.0)  # 20 dense frames
    assert np.array_equal(uniform_frame_indices(600, dense, 32), dense)


class _FakeBatch:
    def __init__(self, indices: List[int]) -> None:
        self._indices = indices

    def asnumpy(self) -> np.ndarray:
        # The first batch finishes last, to check that output order is kept.
        time.sleep(0.05 if self._indices[0] == 0 else 0.0)
        return np.asarray(self._indices)[:, None, None, None]


class _FakeReader:
    instances: List["_FakeReader"] = []

    def __init__(self, path: str, ctx: object, num_threads: int) -> None:
        self.path = path
        self.reads: List[int] = []
        _FakeReader.instances.append(self)

    def get_batch(self, indices: List[int]) -> _FakeBatch:
        self.reads.extend(indices)
        return _FakeBatch(indices)


@pytest.mark.parametrize("num_workers", [1, 3])
def test_frame_batches_keep_order_and_batching(
    monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    monkeypatch.setattr(video, "VideoReader", _FakeReader)
    indices = np.arange(0, 230, 10)  # 23 frames -> batches of 10, 10, 3
    batches = list(
        iter_frame_batches("v.mp4", indices, 10, num_workers, lambda f: f[:, 0, 0, 0])
    )
    assert [len(b) for b in batches] == [10, 10, 3]
    assert np.array_equal(np.concatenate(batches), indices)


def test_one_reader_decodes_all_batches_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Separate readers starting mid-video can decode some frames differently.
    monkeypatch.setattr(video, "VideoReader", _FakeReader)
    monkeypatch.setattr(_FakeReader, "instances", [])
    indices = np.arange(0, 500, 10)
    list(iter_frame_batches("v.mp4", indices, 8, 4, lambda f: f))
    assert len(_FakeReader.instances) == 1
    assert _FakeReader.instances[0].reads == indices.tolist()


class _ThreadedFailReader(_FakeReader):
    """Fails like decord's threaded decoder on some videos when num_threads > 1."""

    def __init__(self, path: str, ctx: object, num_threads: int) -> None:
        super().__init__(path, ctx, num_threads)
        self.num_threads = num_threads

    def get_batch(self, indices: List[int]) -> _FakeBatch:
        if self.num_threads > 1:
            raise video.DECORDError("Thread worker: Error sending packet.")
        return super().get_batch(indices)


def test_threaded_decode_failure_falls_back_to_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video, "VideoReader", _ThreadedFailReader)
    monkeypatch.setattr(_FakeReader, "instances", [])
    indices = np.arange(0, 50, 10)

    def run(threads: int) -> np.ndarray:
        batches = iter_frame_batches("v.mp4", indices, 2, 4, lambda f: f, threads)
        return np.concatenate(list(batches))[:, 0, 0, 0]

    frames = video.with_decode_fallback(run, 4)
    assert frames.tolist() == indices.tolist()
    assert [r.num_threads for r in _FakeReader.instances] == [4, 1]


def test_fallback_reraises_when_one_thread_also_fails() -> None:
    def run(threads: int) -> None:
        raise video.DECORDError("broken file")

    with pytest.raises(video.DECORDError):
        video.with_decode_fallback(run, 4)
