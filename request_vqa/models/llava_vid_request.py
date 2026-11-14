"""LLaVA-Video with ReQuest re-thinking, as an lmms-eval model.

Each multiple-choice question is answered in up to two passes:

1. First thinking: answer from K uniformly sampled frames and measure the
   normalized entropy of the answer-option probabilities.
2. Re-thinking, only when that entropy exceeds the length-aware threshold:
   score every frame at 1 fps with the question-aware selector, pick K frames
   with entropy-adaptive greedy NMS, and answer again from those frames.
"""

import copy
import json
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model.language_model.llava_qwen import LlavaQwenConfig
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from request_vqa.answer_log import load_answers
from request_vqa.benchmarks import OPTION_LETTERS, num_options, question_with_options
from request_vqa.blip_encoder import BlipFusionEncoder
from request_vqa.sampling import adaptive_greedy_nms, nms_interval
from request_vqa.selector import load_selector, score_frames
from request_vqa.uncertainty import (
    normalized_entropy,
    routing_threshold,
    spacing_scale,
)
from request_vqa.video import (
    dense_frame_indices,
    iter_frame_batches,
    uniform_frame_indices,
    with_decode_fallback,
)

# Frames each decode worker reads and preprocesses at once. Kept small so that
# memory stays low for high-resolution videos (a 2560x1920 frame is ~15 MB).
DECODE_CHUNK_FRAMES = 16

AutoConfig.register("llava_llama", LlavaConfig)
AutoConfig.register("llava_qwen", LlavaQwenConfig)


@register_model("llava_vid_request")
class LlavaVidReQuest(lmms):
    """LLaVA-Video that re-selects frames for uncertain questions (ReQuest).

    Defaults are the paper's settings. Per-benchmark values (``tau0`` and the
    selector checkpoint) are set in ``scripts/eval.sh``.
    """

    def __init__(
        self,
        selector_path: str,
        pretrained: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        conv_template: str = "qwen_1_5",
        max_frames_num: int = 32,
        tau0: float = 0.55,
        gamma_len: float = 0.3,
        max_len_frames: int = 3600,
        entropy_min: float = 0.45,
        entropy_max: float = 1.0,
        s_min: float = 0.2,
        rho: float = 0.07,
        dense_fps: float = 1.0,
        blip_model: str = "Salesforce/blip-itm-base-coco",
        blip_batch_size: int = 128,
        selector_batch_size: int = 256,
        decode_workers: int = 4,
        log_dir: Optional[str] = None,
        resume: bool = False,
        torch_dtype: str = "float16",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        attn_implementation: str = "sdpa",
        use_cache: bool = True,
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_mode: str = "average",
        mm_resampler_location: str = "before",
        mm_newline_position: str = "grid",
        **kwargs: Any,
    ) -> None:
        """Load LLaVA-Video, the BLIP encoder, and the frame selector.

        Args:
            selector_path: Selector checkpoint for the target benchmark.
            pretrained: LLaVA-Video checkpoint.
            conv_template: LLaVA conversation template.
            max_frames_num: Frames K given to the MLLM in each pass.
            tau0: Base routing threshold (Video-MME 0.55, MLVU 0.45,
                LongVideoBench 0.3).
            gamma_len: Weight of the video-length term in the threshold.
            max_len_frames: Video length N_max (in 1 fps frames) at which the
                length term saturates.
            entropy_min: Entropy u_min below which NMS keeps uniform spacing.
            entropy_max: Entropy u_max at which spacing reaches ``s_min``.
            s_min: Smallest NMS spacing coefficient.
            rho: Exponent of the entropy-to-spacing mapping.
            dense_fps: Frame rate at which candidate frames are scored.
            blip_model: BLIP ITM checkpoint that produces the selector input.
            blip_batch_size: Frames encoded by BLIP at once.
            selector_batch_size: Frames scored by the selector at once.
            decode_workers: CPU threads that decode (inside one video reader)
                and preprocess frames for BLIP while the GPU encodes earlier
                frames.
            log_dir: If set, per-question routing and selection records are
                written to ``<log_dir>/request_rank<rank>.jsonl``.
            resume: Reuse the answers already recorded in ``log_dir`` (for
                example after an interrupted run) instead of starting over.
        """
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected model_args: {sorted(kwargs)}")
        if not entropy_min < entropy_max:
            raise ValueError("entropy_min must be smaller than entropy_max.")
        resume = str(resume).lower() in ("true", "1")
        if resume and not log_dir:
            raise ValueError("resume=True needs log_dir.")

        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))]
        )
        if accelerator.num_processes == 1 and device_map in ("auto", "balanced_low_0"):
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        overwrite_config = {
            "mm_spatial_pool_stride": int(mm_spatial_pool_stride),
            "mm_spatial_pool_mode": mm_spatial_pool_mode,
            "mm_pooling_position": mm_resampler_location,
            "mm_newline_position": mm_newline_position,
            "delay_load": False,
        }
        self._tokenizer, self._model, self._image_processor, self._max_length = (
            load_pretrained_model(
                pretrained,
                None,
                get_model_name_from_path(pretrained),
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                overwrite_config=overwrite_config,
                attn_implementation=attn_implementation,
            )
        )
        self._config = self._model.config
        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                self._tokenizer.pad_token_id = 151643
        self._model.eval()
        self._model.tie_weights()
        self._setup_parallelism(accelerator, int(batch_size), device_map)

        self.torch_dtype = torch_dtype
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.batch_size_per_gpu = int(batch_size)
        self.max_frames_num = int(max_frames_num)
        self.tau0 = float(tau0)
        self.gamma_len = float(gamma_len)
        self.max_len_frames = int(max_len_frames)
        self.entropy_min = float(entropy_min)
        self.entropy_max = float(entropy_max)
        self.s_min = float(s_min)
        self.rho = float(rho)
        self.dense_fps = float(dense_fps)
        self.blip_batch_size = int(blip_batch_size)
        self.selector_batch_size = int(selector_batch_size)
        self.decode_workers = int(decode_workers)

        self.selector = load_selector(selector_path, self._device)
        self.blip = BlipFusionEncoder(blip_model, self._device)

        self._log_path = None
        self._resumed: Dict[Tuple[str, int], str] = {}
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._log_path = os.path.join(log_dir, f"request_rank{self.rank}.jsonl")
            if resume:
                self._resumed = load_answers(log_dir)
                eval_logger.info(f"Resuming: {len(self._resumed)} answers in {log_dir}")
            else:
                open(self._log_path, "w").close()

    def _setup_parallelism(
        self, accelerator: Accelerator, batch_size: int, device_map: str
    ) -> None:
        if accelerator.num_processes > 1:
            supported = (
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            )
            if accelerator.distributed_type not in supported:
                raise ValueError("Only DDP, FSDP and DeepSpeed are supported.")
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True,
                    train_micro_batch_size_per_gpu=batch_size,
                    train_batch_size=batch_size * accelerator.num_processes,
                )
            if accelerator.distributed_type == DistributedType.MULTI_GPU:
                self._model = accelerator.prepare_model(
                    self._model, evaluation_mode=True
                )
            else:
                self._model = accelerator.prepare(self._model)
            self.accelerator = accelerator
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        elif device_map == "auto":
            self._rank, self._world_size = 0, 1
        else:
            self._model.to(self._device)
            self._rank, self._world_size = 0, 1

    @property
    def config(self) -> Any:
        """Return the LLaVA model config."""
        return self._config

    @property
    def tokenizer(self) -> Any:
        """Return the LLaVA tokenizer."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Return the LLaVA model, unwrapped from Accelerate if needed."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self) -> int:
        """Return the end-of-text token id."""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Return the maximum context length."""
        return self._max_length

    @property
    def batch_size(self) -> int:
        """Return the per-GPU batch size."""
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        """Return the device this process runs on."""
        return self._device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Answer each request with first thinking and, if needed, re-thinking."""
        # With several GPUs, lmms-eval repeats a request on some GPUs so that all
        # get the same number; a repeated request reuses the earlier answer, as
        # does a request answered before a resumed run.
        answers = dict(self._resumed)
        responses = []
        args = [request.args for request in requests]
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in tqdm(
            args, disable=self.rank != 0, desc="ReQuest"
        ):
            key = (task, doc_id)
            if key not in answers:
                doc = self.task_dict[task][split][doc_id]
                video_path = doc_to_visual(doc)[0]
                answers[key], record = self._answer(
                    contexts, dict(gen_kwargs), doc, task, video_path
                )
                self._write_log(
                    {"task": task, "doc_id": doc_id, **record, "answer": answers[key]}
                )
            responses.append(answers[key])
        return responses

    def _answer(
        self,
        contexts: str,
        gen_kwargs: Dict[str, Any],
        doc: Dict[str, Any],
        task: str,
        video_path: str,
    ) -> Tuple[str, Dict[str, Any]]:
        reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        dense = dense_frame_indices(len(reader), reader.get_avg_fps(), self.dense_fps)
        first = uniform_frame_indices(len(reader), dense, self.max_frames_num)
        first_answer, option_probs = self._generate(
            reader.get_batch(first.tolist()).asnumpy(),
            contexts,
            gen_kwargs,
            num_options(doc, task),
        )
        entropy = normalized_entropy(option_probs)
        threshold = routing_threshold(
            len(dense), self.tau0, self.gamma_len, self.max_len_frames
        )
        record: Dict[str, Any] = {
            "num_frames": len(dense),
            "entropy": entropy,
            "threshold": threshold,
            "rethink": entropy > threshold,
            "first_answer": first_answer,
        }
        if entropy <= threshold:
            return first_answer, record

        scores = self._score_frames(video_path, dense, question_with_options(doc, task))
        scale = spacing_scale(
            entropy, self.s_min, self.rho, self.entropy_min, self.entropy_max
        )
        interval = nms_interval(len(dense), self.max_frames_num, scale)
        selected = adaptive_greedy_nms(scores, self.max_frames_num, interval)
        frames = reader.get_batch(dense[selected].tolist()).asnumpy()
        final_answer, _ = self._generate(frames, contexts, gen_kwargs, None)
        record.update(nms_interval=interval, selected_frames=selected.tolist())
        return final_answer, record

    def _score_frames(
        self, video_path: str, dense: np.ndarray, question: str
    ) -> np.ndarray:
        def run(decode_threads: int) -> np.ndarray:
            pixel_chunks = iter_frame_batches(
                video_path,
                dense,
                DECODE_CHUNK_FRAMES,
                self.decode_workers,
                self.blip.preprocess,
                decode_threads=decode_threads,
            )
            fused = self.blip.encode(pixel_chunks, question, self.blip_batch_size)
            return score_frames(self.selector, fused, self.selector_batch_size)

        return with_decode_fallback(run, self.decode_workers)

    def _generate(
        self,
        frames: np.ndarray,
        contexts: str,
        gen_kwargs: Dict[str, Any],
        option_count: Optional[int],
    ) -> Tuple[str, Optional[List[float]]]:
        """Answer from ``frames``.

        If ``option_count`` is given, also return the probabilities of the
        option letters A, B, ... at the first generated token.
        """
        video = self._image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        video = video.bfloat16() if self.torch_dtype == "bfloat16" else video.half()

        image_token = DEFAULT_IMAGE_TOKEN
        if self.model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], f"{image_token}\n{contexts}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if "llama_3" in self.conv_template:
            pad_id = 0
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        temperature = gen_kwargs.get("temperature", 0)

        with torch.inference_mode():
            output = self.model.generate(
                inputs=input_ids,
                images=[video],
                attention_mask=input_ids.ne(pad_id).long(),
                modalities="video",
                use_cache=self.use_cache,
                stopping_criteria=[
                    KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
                ],
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=gen_kwargs.get("top_p"),
                num_beams=gen_kwargs.get("num_beams", 1),
                max_new_tokens=gen_kwargs.get("max_new_tokens", 1024),
                return_dict_in_generate=True,
                output_scores=option_count is not None,
            )
        answer = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0].strip()
        eval_logger.debug(f"Question: {contexts}\nAnswer: {answer}")
        if option_count is None:
            return answer, None

        letters = OPTION_LETTERS[:option_count]
        letter_ids = self.tokenizer.convert_tokens_to_ids(list(letters))
        probs = torch.softmax(output.scores[0], dim=-1)[0]
        return answer, probs[letter_ids].float().tolist()

    def _write_log(self, record: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not supported: ReQuest targets generation-based multiple choice."""
        raise NotImplementedError("llava_vid_request supports generate_until only.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Not supported."""
        raise NotImplementedError("llava_vid_request does not support multi-round.")
