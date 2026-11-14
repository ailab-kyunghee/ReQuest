"""Frame index helpers and frame loading shared by both passes."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Deque, Iterator, Optional, TypeVar

import numpy as np
from decord import VideoReader, cpu
from decord._ffi.base import DECORDError

T = TypeVar("T")


def dense_frame_indices(
    num_video_frames: int, video_fps: float, sample_fps: float
) -> np.ndarray:
    """Return indices of the frames sampled at ``sample_fps`` (1 fps by default)."""
    step = max(1, round(video_fps / sample_fps))
    return np.arange(0, num_video_frames, step)


def uniform_frame_indices(
    num_video_frames: int, dense_indices: np.ndarray, num_frames: int
) -> np.ndarray:
    """Return the frames for the first-thinking pass.

    Matches lmms-eval's LLaVA-Video loader: K evenly spaced frames over the
    whole video, or every dense frame when there are at most K of them.
    """
    if len(dense_indices) > num_frames:
        return np.linspace(0, num_video_frames - 1, num_frames, dtype=int)
    return dense_indices


def iter_frame_batches(
    video_path: str,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    transform: Callable[[np.ndarray], T],
    decode_threads: Optional[int] = None,
) -> Iterator[T]:
    """Yield ``transform(frames)`` for consecutive batches of ``indices``, in order.

    One video reader decodes all batches in order, in a background thread,
    using ``decode_threads`` decoder threads (default ``num_workers``).
    Several readers that each start in the middle of a video can decode some
    frames differently, depending on which reader reads which batch.
    ``num_workers`` threads apply ``transform`` ahead of the consumer, so CPU
    work overlaps with GPU work. At most ``num_workers + 1`` batches are
    pending at once, which bounds memory.
    """
    num_workers = max(1, num_workers)
    threads = num_workers if decode_threads is None else max(1, decode_threads)
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=threads)

    def decode(batch: np.ndarray) -> np.ndarray:
        return reader.get_batch(batch.tolist()).asnumpy()

    def load(decoded: "Future[np.ndarray]") -> T:
        return transform(decoded.result())

    with ThreadPoolExecutor(1) as decoder, ThreadPoolExecutor(num_workers) as pool:
        pending: Deque[Future] = deque()
        for start in range(0, len(indices), batch_size):
            decoded = decoder.submit(decode, indices[start : start + batch_size])
            pending.append(pool.submit(load, decoded))
            if len(pending) > num_workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def with_decode_fallback(run: Callable[[int], T], decode_threads: int) -> T:
    """Return ``run(decode_threads)``, or ``run(1)`` if threaded decoding fails.

    decord's multi-threaded decoder fails on some videos (for example some
    Ego4D clips) that decode fine with one thread. Where both work, they give
    identical frames. ``run`` must decode the whole video again from the start.
    """
    try:
        return run(decode_threads)
    except DECORDError:
        if decode_threads <= 1:
            raise
        return run(1)
