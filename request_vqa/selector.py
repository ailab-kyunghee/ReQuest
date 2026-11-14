"""Question-aware frame selector."""

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

# The research checkpoints also hold an unused text branch.
_UNUSED_KEY_PREFIXES = ("t_proj.", "t_pos.")


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to a token sequence."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``tokens`` of shape [B, T, D]."""
        pe = self.pe[: tokens.size(1)].to(dtype=tokens.dtype, device=tokens.device)
        return tokens + pe.unsqueeze(0)


class QuestionAwareSelector(nn.Module):
    """Scores a frame from its BLIP frame-question fused tokens.

    A learnable score token is prepended to the fused tokens, a Transformer
    encoder aggregates the sequence, and the score token's final state is
    mapped to a scalar contribution score.
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Module names match the released checkpoints.
        self.v_proj = nn.Linear(input_dim, d_model)
        self.v_pos = PositionalEncoding(d_model)
        self.score_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, ff_dim, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, fused_tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return one score per input: [B, T, input_dim] -> [B].

        Args:
            fused_tokens: BLIP fused question tokens of a frame or segment.
            padding_mask: Optional [B, T] bool mask, True for padding tokens
                that the encoder must ignore. Training batches inputs of
                different lengths with it; inference scores one question at a
                time and needs no padding.
        """
        tokens = self.v_pos(self.v_proj(fused_tokens))
        batch = tokens.size(0)
        score_token = self.score_token.to(tokens.dtype).expand(batch, 1, -1)
        sequence = torch.cat([score_token, tokens], dim=1)
        if padding_mask is None:
            padding_mask = torch.zeros(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
        score_slot = torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device)
        mask = torch.cat([score_slot, padding_mask.to(tokens.device)], dim=1)
        hidden = self.encoder(sequence, src_key_padding_mask=mask)
        return self.head(hidden[:, 0]).squeeze(-1)


def load_selector(
    path: Union[str, Path], device: Union[str, torch.device]
) -> QuestionAwareSelector:
    """Load a selector checkpoint (a state dict) in eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith(_UNUSED_KEY_PREFIXES)}
    selector = QuestionAwareSelector()
    selector.load_state_dict(state, strict=True)
    return selector.to(device).eval()


@torch.no_grad()
def score_frames(
    selector: QuestionAwareSelector, fused_tokens: torch.Tensor, batch_size: int
) -> np.ndarray:
    """Score every frame of ``fused_tokens`` [N, T, D] in batches of frames."""
    device = next(selector.parameters()).device
    scores = [
        selector(chunk.to(device=device, dtype=torch.float32))
        for chunk in fused_tokens.split(batch_size)
    ]
    return torch.cat(scores).cpu().numpy()
