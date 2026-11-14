"""Train the question-aware selector on the pseudo-label dataset.

Usage:
    python scripts/train_selector.py --data_dir DATA --features_dir FEATURES \
        --output_dir runs/selector [--epochs 20]
    torchrun --nproc_per_node 2 scripts/train_selector.py ...   # several GPUs

DATA holds samples.parquet; FEATURES holds the output of
extract_blip_features.py. A checkpoint ``epoch_<k>.pth`` is saved after every
epoch and loads directly as ``selector_path`` of ``llava_vid_request``.
``last.pt`` also keeps the optimizer, scheduler and random states, and
``--resume`` continues an interrupted run from the last finished epoch exactly
as if it had not stopped (with the same number of GPUs).

The learning rate follows a cosine schedule over ``--schedule_epochs`` (300)
even when training stops earlier, so epoch k sees the same learning rate as in
the paper's run. The released checkpoints are epoch 20 (Video-MME), 2
(LongVideoBench) and 1 (MLVU) of one run.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from request_vqa.selector import QuestionAwareSelector
from request_vqa.training.data import (
    ContiguousBatchSampler,
    PseudoLabelDataset,
    collate,
    feature_path,
)
from request_vqa.training.losses import selector_loss

REFERENCE_BATCH = 512  # the learning rate is given for this global batch size


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20, help="epochs to train")
    parser.add_argument("--schedule_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512, help="per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rank_weight", type=float, default=0.3)
    parser.add_argument("--rank_delta", type=float, default=0.05)
    parser.add_argument("--smoothl1_beta", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue last.pt")
    parser.add_argument("--device", default=None, help="default: cuda if available")
    return parser.parse_args(argv)


def _random_state() -> Dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_random_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(obj: object, path: Path) -> None:
    """Save atomically, so an interruption never leaves a broken file."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        default = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(args.device or default)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = pd.read_parquet(Path(args.data_dir) / "samples.parquet")
    available = [
        feature_path(args.features_dir, q).exists() for q in samples["question_id"]
    ]
    missing = len(samples) - sum(available)
    samples = samples[available].reset_index(drop=True)
    dataset = PseudoLabelDataset(samples, args.features_dir)
    sampler = ContiguousBatchSampler(len(dataset), args.batch_size, world_size, rank)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = QuestionAwareSelector().to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])
    lr = args.lr * args.batch_size * world_size / REFERENCE_BATCH
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.schedule_epochs, eta_min=args.min_lr
    )

    output = Path(args.output_dir)
    first_epoch = 1
    if args.resume and (output / "last.pt").exists():
        last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        (model.module if distributed else model).load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        random_file = output / f"random_rank{rank}_epoch{last['epoch']}.pt"
        _set_random_state(torch.load(random_file, weights_only=False))
        first_epoch = last["epoch"] + 1
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        config = dict(vars(args), samples=len(dataset), skipped=missing)
        (output / "config.json").write_text(json.dumps(config, indent=2))
        print(
            f"{len(dataset)} samples ({missing} skipped: no features), "
            f"epochs {first_epoch}-{args.epochs}",
            flush=True,
        )

    for epoch in range(first_epoch, args.epochs + 1):
        sampler.set_epoch(epoch - 1)
        model.train()
        start, total, steps = time.time(), 0.0, 0
        for batch in loader:
            scores = model(
                batch["tokens"].to(device, torch.float32, non_blocking=True),
                batch["padding_mask"].to(device, non_blocking=True),
            )
            loss = selector_loss(
                scores,
                batch["labels"].to(device),
                batch["question_ids"],
                rank_weight=args.rank_weight,
                smoothl1_beta=args.smoothl1_beta,
                rank_delta=args.rank_delta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
        scheduler.step()
        if distributed:
            dist.barrier()
        # Saved per epoch, so they always match the epoch in last.pt.
        _save(_random_state(), output / f"random_rank{rank}_epoch{epoch}.pt")
        if rank == 0:
            state = (model.module if distributed else model).state_dict()
            _save(state, output / f"epoch_{epoch}.pth")
            last = dict(
                epoch=epoch,
                model=state,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            _save(last, output / "last.pt")
            log = dict(
                epoch=epoch,
                loss=total / max(steps, 1),
                steps=steps,
                lr=optimizer.param_groups[0]["lr"],
                seconds=time.time() - start,
            )
            with open(output / "train_log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
