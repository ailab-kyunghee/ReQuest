"""Extract per-question BLIP segment features for selector training.

For every question of the pseudo-label dataset, BLIP fuses the question with
each 1 fps frame of its video, and the fused question tokens are averaged over
the frames of each segment. The result, float16 [num_segments, T, 768], is
saved as ``<output_dir>/<question_id>.safetensors``. The vision encoder runs
once per video and the text encoder once per question.

Usage:
    python scripts/extract_blip_features.py --data_dir DATA --video_root VIDEOS \
        --output_dir FEATURES [--num_shards 4 --shard 0]

DATA holds questions.parquet and segments.parquet of the pseudo-label
dataset; VIDEOS is the LLaVA-Video-178K root. Each question's file is written
atomically as soon as its video is done, and finished questions are skipped,
so an interrupted run resumes by running the same command again. Videos that
cannot be read are listed in ``failed_shard<k>.txt`` and retried on the next
run.
"""

import argparse
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch  # must be imported before decord
from decord import VideoReader, cpu
from safetensors.torch import save_file

from request_vqa.blip_encoder import BlipFusionEncoder
from request_vqa.training.data import (
    FEATURE_KEY,
    align_segments,
    feature_path,
    segment_means,
)
from request_vqa.video import (
    dense_frame_indices,
    iter_frame_batches,
    with_decode_fallback,
)

DECODE_CHUNK_FRAMES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--blip_model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--batch_size", type=int, default=128, help="BLIP frames")
    parser.add_argument("--decode_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def encode_video(
    blip: BlipFusionEncoder,
    path: str,
    questions: List[str],
    args: argparse.Namespace,
) -> Tuple[List[torch.Tensor], int]:
    """Return the fused tokens of each question and the number of 1 fps frames."""
    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    dense = dense_frame_indices(len(reader), reader.get_avg_fps(), 1.0)
    del reader

    def run(decode_threads: int) -> List[torch.Tensor]:
        chunks = iter_frame_batches(
            path,
            dense,
            DECODE_CHUNK_FRAMES,
            args.decode_workers,
            blip.preprocess,
            decode_threads=decode_threads,
        )
        return blip.encode_questions(chunks, questions, args.batch_size)

    return with_decode_fallback(run, args.decode_workers), len(dense)


def main() -> None:
    args = parse_args()
    data = Path(args.data_dir)
    questions = pd.read_parquet(data / "questions.parquet")
    segments = pd.read_parquet(data / "segments.parquet").set_index("video_id")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Temporary files of this shard left by an interrupted run.
    temporary_suffix = f".tmp-shard{args.shard}"
    for leftover in output.glob(f"*{temporary_suffix}"):
        leftover.unlink()
    failed_log = output / f"failed_shard{args.shard}.txt"

    shard = questions[
        questions["video_id"].isin(
            sorted(questions["video_id"].unique())[args.shard :: args.num_shards]
        )
    ]
    todo_all = shard[
        [not feature_path(output, q).exists() for q in shard["question_id"]]
    ]
    videos = sorted(todo_all["video_id"].unique())
    print(
        f"shard {args.shard}: {len(todo_all)} of {len(shard)} questions left, "
        f"in {len(videos)} videos",
        flush=True,
    )
    by_video = todo_all.groupby("video_id")
    blip = BlipFusionEncoder(args.blip_model, args.device)
    stats: Counter = Counter()
    start = time.time()
    for done, video_id in enumerate(videos, 1):
        todo = by_video.get_group(video_id)
        path = os.path.join(args.video_root, todo["video"].iloc[0])
        try:
            fused, num_frames = encode_video(
                blip, path, todo["question"].tolist(), args
            )
        except Exception as error:  # unreadable or missing video
            stats["failed videos"] += 1
            print(f"FAILED {video_id} {path}: {error}", flush=True)
            with open(failed_log, "a") as f:
                f.write(f"{video_id}\t{path}\t{error}\n")
        else:
            frame_segments = segments.loc[video_id, "frame_segments"]
            stats["segment ids resized"] += len(frame_segments) != num_frames
            ids = align_segments(frame_segments, num_frames)
            for question_id, tokens in zip(todo["question_id"], fused):
                target = feature_path(output, question_id)
                temporary = target.with_name(target.name + temporary_suffix)
                save_file({FEATURE_KEY: segment_means(tokens, ids)}, str(temporary))
                os.replace(temporary, target)
                stats["questions written"] += 1
        if done % 100 == 0 or done == len(videos):
            rate = done / (time.time() - start)
            eta_h = (len(videos) - done) / max(rate, 1e-9) / 3600
            progress = f"{done}/{len(videos)} videos, {dict(stats)}"
            print(f"{progress}, ETA {eta_h:.1f} h", flush=True)


if __name__ == "__main__":
    main()
