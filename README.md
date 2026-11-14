# ReQuest: Rethinking-based Question-Aware Frame Selection for Long-Form Video QA (ECCV 2026)

Official codebase for the ECCV 2026 paper:  
**"ReQuest: Rethinking-based Question-Aware Frame Selection for Long-Form Video QA"**  
[[Paper]](https://arxiv.org/abs/2607.01737)

---

## 🧠 Overview

**ReQuest** is a plug-and-play frame selection framework for long-form video question answering.
It changes which frames the MLLM sees, not the MLLM itself, and is packaged here as an
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) plugin on top of LLaVA-Video.

Each multiple-choice question is answered in up to two passes
([`llava_vid_request.py`](request_vqa/models/llava_vid_request.py)):

1. **First thinking.** LLaVA-Video answers from 32 uniformly sampled frames. The normalized
   entropy `u` of the answer-option probabilities measures its uncertainty.
2. **Re-thinking routing** ([`uncertainty.py`](request_vqa/uncertainty.py)). If
   `u > tau0 - gamma_len * min(1, N / N_max)`, where `N` is the number of frames at 1 fps, the
   question is re-thought. Otherwise the first answer is final.
3. **Question-aware selection** ([`blip_encoder.py`](request_vqa/blip_encoder.py),
   [`selector.py`](request_vqa/selector.py)). Every 1 fps frame is fused with the question by
   BLIP, and the selector scores each frame.
4. **Adaptive NMS** ([`sampling.py`](request_vqa/sampling.py)). Greedy NMS picks 32 frames
   with suppression interval `round(s(u) * N / 32)`, so higher uncertainty gives denser sampling.
   If suppression runs out of candidates, NMS repeats over the remaining frames. LLaVA-Video
   then answers again from the selected frames.

---

## 📁 Directory Structure

<pre>
📁 ReQuest
├── request_vqa/                       # lmms-eval plugin
│   ├── models/llava_vid_request.py    # LLaVA-Video + ReQuest (first thinking → routing → selection → re-thinking)
│   ├── uncertainty.py                 # Normalized entropy and re-thinking routing
│   ├── blip_encoder.py                # BLIP question-frame fusion
│   ├── selector.py                    # Question-aware frame selector
│   ├── sampling.py                    # Uncertainty-adaptive NMS
│   ├── video.py                       # Frame sampling and decoding
│   ├── benchmarks.py                  # Benchmark-specific question text and options
│   ├── answer_log.py                  # Per-question log, used to resume runs
│   ├── tasks/                         # videomme_request and mlvu_request task configs
│   └── training/                      # Pseudo-label dataset and selector loss
├── scripts/
│   ├── eval.sh                        # Evaluation on Video-MME, LongVideoBench and MLVU
│   ├── extract_features.sh            # BLIP feature extraction for selector training
│   ├── extract_blip_features.py
│   ├── train.sh                       # Selector training
│   └── train_selector.py
├── tests/                             # Unit tests
├── checkpoints/                       # Selector checkpoints (to be downloaded)
│   ├── request_selector_videomme.pth
│   ├── request_selector_longvideobench.pth
│   └── request_selector_mlvu.pth
├── data/                              # Selector training data (to be downloaded)
│   ├── pseudo_labels/                 # Pseudo-labels (samples, questions, segments)
│   ├── LLaVA-Video-178K/              # LLaVA-Video-178K videos
│   └── features/                      # Output of extract_features.sh
├── pyproject.toml
└── README.md
</pre>

---

## 🛠️ Setup

We recommend using a Conda environment:

```bash
conda create --name ReQuest python=3.12
conda activate ReQuest
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "llava @ git+https://github.com/LLaVA-VL/LLaVA-NeXT.git@b42941ceba259d5df18f8df8193a3897296a0449"
pip install -e .
```

`pip install -e .` installs lmms-eval 0.4.0, transformers 4.39.3, accelerate 0.29.3,
datasets 2.19.0 and pyarrow 19.0.1, and requires torch 2.5.1. These are the versions we
tested, with Python 3.12 and decord 0.6.0, on one to four 48 GB GPUs (RTX A6000). Newer
lmms-eval and transformers releases are untested, and newer pyarrow needs numpy 2, which
lmms-eval 0.4.0 does not allow.

lmms-eval downloads the benchmarks, and LLaVA-Video-7B-Qwen2 and BLIP are downloaded
from Hugging Face, the first time they are used.

---

## 🔗 Pretrained Models

We provide one selector checkpoint per benchmark:

| Model | Benchmark | Download Link | Save Path |
|-------|-----------|---------------|-----------|
| ReQuest selector | Video-MME | [Hugging Face](https://huggingface.co/suyong-3/ReQuest/tree/main) | `checkpoints/request_selector_videomme.pth` |
| ReQuest selector | LongVideoBench | [Hugging Face](https://huggingface.co/suyong-3/ReQuest/tree/main) | `checkpoints/request_selector_longvideobench.pth` |
| ReQuest selector | MLVU | [Hugging Face](https://huggingface.co/suyong-3/ReQuest/tree/main) | `checkpoints/request_selector_mlvu.pth` |

```bash
huggingface-cli download suyong-3/ReQuest --include "request_selector_*.pth" --local-dir checkpoints
```

Make sure to place the downloaded `.pth` files in `checkpoints/` as shown above.

---

## 📂 Pseudo-Label Download

The pseudo-labels used to train the selector are provided on Hugging Face. They are needed
only to train the selector, not to evaluate it.

| Data | Download Link | Save Path |
|------|---------------|-----------|
| ReQuest pseudo-labels | [Hugging Face](https://huggingface.co/datasets/suyong-3/ReQuest/tree/main) | `data/pseudo_labels/*` |
| LLaVA-Video-178K videos | [Hugging Face](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) | `data/LLaVA-Video-178K/*` |

```bash
huggingface-cli download suyong-3/ReQuest --repo-type dataset --local-dir data/pseudo_labels
```

| File | Rows | Columns |
|------|------|---------|
| `samples.parquet` | 803,245 | `question_id`; `segments`, a set of segment ids of the video; `label`, the selector's target score for those segments |
| `questions.parquet` | 83,421 | `question_id`; `video_id`; `video`, the path under the LLaVA-Video-178K root; `question`, the question with its options |
| `segments.parquet` | 33,406 | `video_id`; `frame_segments`, the segment id of every 1 fps frame |

> 📌 *Extract the LLaVA-Video-178K videos so that the `video` paths in `questions.parquet`
> resolve under `data/LLaVA-Video-178K/`. The code that generated the pseudo-labels is not
> part of this repository.*

---

## 🧩 Question-Aware Selector Training

**1. Extract BLIP features.** BLIP fuses each question with every 1 fps frame of its video,
and the fused tokens are averaged within each segment. Each question gets one file in
`data/features/` (about 155 GB in total).

```bash
bash scripts/extract_features.sh 4    # 4 processes in parallel
```

Decoding the frames is the slow part. With 4 processes on 16 CPU cores and one GPU, it takes
about 15 hours. Finished questions are skipped, so run the same command again to resume.

**2. Train the selector.**

```bash
bash scripts/train.sh                        # 1 GPU, batch 512
bash scripts/train.sh 2 --batch_size 256     # 2 GPUs, same global batch
```

One epoch takes about 8.5 minutes on one RTX A6000. A checkpoint `runs/selector/epoch_<k>.pth`
is saved after every epoch, and running the same command again continues the run exactly.
The learning rate scales with the global batch size, and other numbers of GPUs see the
data in a different order than one GPU. Evaluate a checkpoint with
`SELECTOR=runs/selector/epoch_20.pth bash scripts/eval.sh videomme`.

The released checkpoints are epochs 20 (Video-MME), 2 (LongVideoBench) and 1 (MLVU) of the
run behind the paper. That run used an earlier version of the feature extraction, which
stored one set of fused tokens per video instead of per question and averaged some segments
over the wrong frames. This code fixes both. Retraining with it gives the results in
[Evaluation Metrics](#-evaluation-metrics).

---

## 🚀 Evaluation

```bash
bash scripts/eval.sh videomme                                    # 1 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh videomme 4     # 4 GPUs
bash scripts/eval.sh mlvu 1 --limit 20                           # quick check
```

| Benchmark | Argument | `tau0` | lmms-eval task |
|-----------|----------|--------|----------------|
| Video-MME (w/o subtitles) | `videomme` | 0.55 | `videomme_request` |
| LongVideoBench (val) | `longvideobench` | 0.30 | `longvideobench_val_v` |
| MLVU (dev) | `mlvu` | 0.45 | `mlvu_request` |

The second argument, optional, is the number of GPUs (default 1). Each GPU holds a full
copy of the models and answers a share of the questions. Any further arguments go to
lmms-eval. `SELECTOR`, `OUTPUT` and `PORT` environment variables override the checkpoint
path, the output directory and the port used between GPU processes. `EXTRA_MODEL_ARGS` adds
model arguments, e.g. `EXTRA_MODEL_ARGS=decode_workers=8`.

To continue an interrupted run, run the same command again with
`EXTRA_MODEL_ARGS=resume=True`. Questions already answered in the log are skipped.
For every question, `outputs/<benchmark>/request_log/request_rank*.jsonl` (one file per GPU)
records the entropy, the routing threshold, whether it was re-thought, the selected
frames, and the final answer.

Re-thinking decodes and encodes every 1 fps frame, which is mostly CPU work, so a
re-thought question on a 45-minute video takes one to three minutes.
`decode_workers` (default 4) sets how many CPU threads per GPU decode and preprocess
frames ahead of BLIP. One video reader decodes the frames in order, because separate
readers can decode some frames differently. Each pending batch of frames takes memory,
so raise it only if you have spare CPU cores and memory.

### Hyperparameters

All are `--model_args` of `llava_vid_request`. The defaults are the paper's values.

| Argument | Default | Meaning |
|----------|---------|---------|
| `max_frames_num` | 32 | Frames K given to the MLLM |
| `tau0` | 0.55 | Base routing threshold (set per benchmark by `eval.sh`) |
| `gamma_len` | 0.3 | Weight of the video-length term |
| `max_len_frames` | 3600 | Length N_max at which the length term saturates |
| `entropy_min`, `entropy_max` | 0.45, 1.0 | Entropy range mapped to NMS spacing |
| `s_min` | 0.2 | Smallest spacing coefficient |
| `rho` | 0.07 | Exponent of the entropy-to-spacing mapping |
| `dense_fps` | 1.0 | Rate at which candidate frames are scored |
| `decode_workers` | 4 | CPU threads per GPU that decode frames for BLIP (speed only) |

---

## 📊 Evaluation Metrics

- **Video-MME** (w/o subtitles): accuracy
- **LongVideoBench** (val): accuracy
- **MLVU** (dev): m-avg, the mean accuracy over its multiple-choice tasks

Evaluation is handled by lmms-eval. `videomme_request` is lmms-eval's Video-MME task with
the post-prompt lmms-eval uses for LLaVA-Video (`The best answer is:`); lmms-eval chooses
prompts by model name, so the stock `videomme` task would give `llava_vid_request` a
different prompt. `mlvu_request` is lmms-eval's `mlvu_dev` scored on the option letter the
answer starts with, as in the paper: the stock task scores an answer such as `C.` as wrong,
which lowers the m-avg by about four points. LongVideoBench uses the stock task.

Results of `scripts/eval.sh` with the released checkpoints, and with a selector retrained
with this code (same epochs):

| Benchmark | Paper | Released checkpoint | Retrained with this code |
|-----------|-------|---------------------|--------------------------|
| Video-MME (w/o subtitles) | 65.6 | 65.19 | 64.59 |
| LongVideoBench (val) | 60.1 | 59.76 | 60.13 |
| MLVU (dev), m-avg | 71.7 | 71.68 | 71.75 |

This code is a cleaned-up version of the code behind the paper, so scores can differ from
the paper by up to about 0.4 points.

---

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{kim2026request,
  title     = {ReQuest: Rethinking-based Question-Aware Frame Selection for Long-Form Video QA},
  author    = {Kim, Minkuk and Yun, Suyong and Kim, Young Tae and Moon, Jinyoung and
               Choi, Jinwoo and Kim, Seong Tae},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## 📝 License

This project is licensed under the [Apache License 2.0](./LICENSE).

---

## 🙏 Acknowledgements

Our framework builds upon [LLaVA-NeXT (LLaVA-Video)](https://github.com/LLaVA-VL/LLaVA-NeXT),
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval),
[BLIP](https://github.com/salesforce/BLIP) and
[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K). We thank the
open-source community!
