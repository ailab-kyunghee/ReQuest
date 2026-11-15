<div align="center">

# ReQuest

### Rethinking-based Question-Aware Frame Selection for Long-Form Video QA

[Minkuk Kim](https://www.linkedin.com/in/minkuk-kim/)<sup>1*</sup>, Suyong Yun<sup>1*</sup>, Young Tae Kim<sup>1</sup>, Jinyoung Moon<sup>2</sup>, [Jinwoo Choi](https://www.linkedin.com/in/jinwoo-choi-36056795/)<sup>1†</sup>, [Seong Tae Kim](https://www.linkedin.com/in/seong-tae-kim-b2b04086/)<sup>1†</sup>

<sup>1</sup>Kyung Hee University &nbsp;&nbsp; <sup>2</sup>Electronics and Telecommunications Research Institute (ETRI)

<sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding authors

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-1F4E8C)](https://eccv.ecva.net/)
[![Project Page](https://img.shields.io/badge/Project-Page-A40F16)](https://geppa.github.io/ReQuest/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.01737-b31b1b)](https://arxiv.org/abs/2607.01737)

</div>

<p align="center">
  <img src="https://geppa.github.io/ReQuest/static/images/teaser.jpg" width="90%" alt="ReQuest overview">
</p>

## News

- **[2026.11]** Code, selector checkpoints, pseudo-labels and BLIP features are released.
- **[2026.06]** ReQuest is accepted to ECCV 2026.

## Overview

ReQuest is a plug-and-play frame selection framework for long-form video question answering.
It changes which frames the MLLM sees, not the MLLM itself. This repository implements it on
LLaVA-Video as an [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) plugin.

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

## Installation

```bash
conda create --name ReQuest python=3.12
conda activate ReQuest
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "llava @ git+https://github.com/LLaVA-VL/LLaVA-NeXT.git@b42941ceba259d5df18f8df8193a3897296a0449"
pip install -e .
huggingface-cli login
```

`pip install -e .` installs lmms-eval 0.4.0, transformers 4.39.3, accelerate 0.29.3,
datasets 2.19.0 and pyarrow 19.0.1. These are the versions we tested, with Python 3.12, on
one to four 48 GB GPUs (RTX A6000); newer lmms-eval and transformers releases are untested.

The benchmarks, LLaVA-Video-7B-Qwen2 and BLIP are downloaded from Hugging Face the first time
they are used. lmms-eval needs a Hugging Face token for this, hence `huggingface-cli login`.
LongVideoBench also requires accepting its terms on its
[dataset page](https://huggingface.co/datasets/longvideobench/LongVideoBench).

## Evaluation

### Checkpoints

We provide one selector checkpoint per benchmark on
[Hugging Face](https://huggingface.co/suyong-3/ReQuest).

```bash
huggingface-cli download suyong-3/ReQuest --include "request_selector_*.pth" --local-dir checkpoints
```

| Benchmark | Checkpoint | `tau0` | lmms-eval task |
|-----------|------------|--------|----------------|
| Video-MME (w/o subtitles) | `checkpoints/request_selector_videomme.pth` | 0.55 | `videomme_request` |
| LongVideoBench (val) | `checkpoints/request_selector_longvideobench.pth` | 0.30 | `longvideobench_val_v` |
| MLVU (dev) | `checkpoints/request_selector_mlvu.pth` | 0.45 | `mlvu_request` |

### Running

```bash
bash scripts/eval.sh videomme                                    # 1 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh videomme 4     # 4 GPUs
bash scripts/eval.sh mlvu 1 --limit 20                           # quick check
```

The first argument is the benchmark (`videomme`, `longvideobench` or `mlvu`) and the second,
optional, is the number of GPUs (default 1). Each GPU holds a full copy of the models and
answers a share of the questions. Further arguments go to lmms-eval. The environment
variables `SELECTOR`, `OUTPUT` and `PORT` override the checkpoint, the output directory and
the port used between GPU processes, and `EXTRA_MODEL_ARGS` adds model arguments, e.g.
`EXTRA_MODEL_ARGS=decode_workers=8`.

For every question, `outputs/<benchmark>/request_log/request_rank*.jsonl` records the
entropy, the routing threshold, whether it was re-thought, the selected frames and the final
answer. To continue an interrupted run, run the same command again with
`EXTRA_MODEL_ARGS=resume=True`; questions already in the log are skipped.

Re-thinking decodes and encodes every 1 fps frame, which is mostly CPU work: a re-thought
question on a 45-minute video takes one to three minutes. `decode_workers` (default 4) sets
the CPU threads per GPU that decode and preprocess frames ahead of BLIP.

### Hyperparameters

All are `--model_args` of `llava_vid_request`. The defaults are the values in the paper.

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

## Training

### Data

The selector is trained on pseudo-labels for questions from LLaVA-Video-178K. We release
the pseudo-labels together with the BLIP features of every question on
[Hugging Face](https://huggingface.co/datasets/suyong-3/ReQuest), so no videos are needed.

```bash
huggingface-cli download suyong-3/ReQuest --repo-type dataset --include "*.parquet" --local-dir data/pseudo_labels
huggingface-cli download suyong-3/ReQuest --repo-type dataset --include "features/*" --local-dir data
for f in data/features/*.tar; do tar -xf "$f" -C data/features && rm "$f"; done
```

| Data | Size | Content |
|------|------|---------|
| `samples.parquet` | 803,245 rows | `question_id`; `segments`, a set of segment ids of the video; `label`, the selector's target score for those segments |
| `questions.parquet` | 83,421 rows | `question_id`; `video_id`; `video`, the source video in LLaVA-Video-178K; `question`, the question with its options |
| `segments.parquet` | 33,406 rows | `video_id`; `frame_segments`, the segment id of every 1 fps frame |
| `features/` | 83,421 files, 164 GB | `<question_id>.safetensors`: BLIP-fused question tokens averaged over the frames of each segment, float16 [segments, tokens, 768] |

### Training

```bash
bash scripts/train.sh                        # 1 GPU, batch 512
bash scripts/train.sh 2 --batch_size 256     # 2 GPUs, same global batch
```

One epoch takes about 8.5 minutes on one RTX A6000. A checkpoint `runs/selector/epoch_<k>.pth`
is saved after every epoch, and running the same command again resumes the run. The learning
rate scales with the global batch size. Evaluate a checkpoint with
`SELECTOR=runs/selector/epoch_20.pth bash scripts/eval.sh videomme`.

The released checkpoints are epochs 20 (Video-MME), 2 (LongVideoBench) and 1 (MLVU).

## Results

| Benchmark | LLaVA-Video-7B + ReQuest |
|-----------|--------------------------|
| Video-MME (w/o subtitles) | 65.6 |
| LongVideoBench (val) | 60.1 |
| MLVU (dev), m-avg | 71.7 |

## Citation

```bibtex
@inproceedings{kim2026request,
  title     = {{ReQuest}: Rethinking-Based Question-Aware Frame
               Selection for Long-Form Video {QA}},
  author    = {Kim, Minkuk and Yun, Suyong and Kim, Young Tae and
               Moon, Jinyoung and Choi, Jinwoo and Kim, Seong Tae},
  booktitle = {Computer Vision -- ECCV 2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {17052},
  pages     = {386--404},
  publisher = {Springer},
  address   = {Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-37235-2_23}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). The pseudo-labels are
derived from [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K),
which is licensed for academic research and education only.

## Acknowledgements

This code builds on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT),
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and
[BLIP](https://github.com/salesforce/BLIP).
