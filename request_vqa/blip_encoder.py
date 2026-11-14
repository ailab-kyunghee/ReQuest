"""BLIP frame-question fused tokens, the selector input."""

from typing import Iterable, Iterator, List, Sequence, Union

import numpy as np
import torch
from transformers import AutoProcessor, BlipForImageTextRetrieval


def rebatch(chunks: Iterable[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    """Regroup chunks of frames into batches of ``batch_size`` frames, in order.

    The last batch may be smaller.
    """
    buffer: List[torch.Tensor] = []
    count = 0
    for chunk in chunks:
        buffer.append(chunk)
        count += len(chunk)
        while count >= batch_size:
            merged = torch.cat(buffer)
            yield merged[:batch_size]
            rest = merged[batch_size:]
            buffer, count = ([rest] if len(rest) else []), len(rest)
    if buffer:
        yield torch.cat(buffer)


class BlipFusionEncoder:
    """Encodes video frames together with a question using BLIP's ITM branch.

    For every frame, BLIP's text encoder cross-attends from the question
    tokens to the frame's visual tokens. The fused states of the question
    tokens (without [CLS] and [SEP]) are the selector input.
    """

    def __init__(self, model_name: str, device: Union[str, torch.device]) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name)
        self.model.to(device).eval()
        tokenizer = self.processor.tokenizer
        self._cls_id = tokenizer.cls_token_id
        self._sep_id = tokenizer.sep_token_id
        max_positions = self.model.text_encoder.config.max_position_embeddings
        self._max_question_tokens = max_positions - 2

    def _question_ids(self, question: str) -> torch.Tensor:
        ids = self.processor.tokenizer(
            question,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_question_tokens,
        )["input_ids"]
        return torch.tensor([[self._cls_id, *ids, self._sep_id]], device=self.device)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """Turn RGB frames [B, H, W, 3] into BLIP pixel values on the CPU."""
        return self.processor(images=list(frames), return_tensors="pt")["pixel_values"]

    def encode(
        self, pixel_chunks: Iterable[torch.Tensor], question: str, batch_size: int
    ) -> torch.Tensor:
        """Return fused tokens [num_frames, num_question_tokens, hidden].

        Args:
            pixel_chunks: Chunks of frames from :meth:`preprocess`, of any size.
            question: Question text, including the answer options.
            batch_size: Frames BLIP encodes at once.

        Returns:
            float16 tensor on the CPU, matching the precision the selector was
            trained with.
        """
        return self.encode_questions(pixel_chunks, [question], batch_size)[0]

    @torch.no_grad()
    def encode_questions(
        self,
        pixel_chunks: Iterable[torch.Tensor],
        questions: Sequence[str],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """Return fused tokens for each question about the same frames.

        The vision encoder runs once per batch of frames, and the text encoder
        once per question. Each result has the shape and precision that
        :meth:`encode` returns for that question alone.
        """
        question_ids = [self._question_ids(question) for question in questions]
        fused: List[List[torch.Tensor]] = [[] for _ in questions]
        for pixel_values in rebatch(pixel_chunks, batch_size):
            pixel_values = pixel_values.to(self.device)
            image_embeds = self.model.vision_model(pixel_values).last_hidden_state
            image_mask = torch.ones(
                image_embeds.shape[:2], dtype=torch.long, device=self.device
            )
            for outputs, input_ids in zip(fused, question_ids):
                ids = input_ids.expand(len(pixel_values), -1)
                hidden = self.model.text_encoder(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_mask,
                    return_dict=True,
                ).last_hidden_state
                outputs.append(hidden[:, 1:-1].to(torch.float16).cpu())
        return [torch.cat(outputs) for outputs in fused]
