"""Differentiable acceptance rate proxy losses for speculative decoding.

Provides alternatives to KL divergence that more directly approximate
the speculative decoding acceptance probability.
"""

import torch
import torch.nn.functional as F
from typing import Optional

EPSILON = 1e-10


def compute_overlap_loss(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int = 50,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor | float]:
    """Top-k overlap loss (Proxy B).

    Maximizes probability mass overlap between target and draft distributions
    over the draft's top-k tokens. Minimizing the negative overlap pushes
    the trainable model toward the other's distribution.

    The overlap at each position is defined as:
        overlap = sum_{x in top_k(draft)} min(p_target(x), p_draft(x))

    This directly approximates the speculative decoding acceptance probability,
    since the acceptance rate under optimal rejection sampling is exactly
    sum_x min(p_target(x), p_draft(x)) over the full vocabulary. Restricting
    to the draft's top-k provides a tight lower bound while enabling efficient
    gradient computation.

    Args:
        target_logits: (batch, seq_len, vocab_size). Detach externally if needed.
        draft_logits: (batch, seq_len, vocab_size). Detach externally if needed.
        attention_mask: (batch, seq_len). 1 for real tokens, 0 for padding.
        top_k: Number of top draft tokens to compute overlap over.
        temperature: Softmax temperature.

    Returns:
        dict with:
            overlap_loss: Negative overlap (tensor, for backprop — minimize this).
            overlap_value: Mean overlap value (detached float, for logging).
    """
    # Shift to align with causal LM convention (predict next token)
    shift_target = target_logits[..., :-1, :] / temperature
    shift_draft = draft_logits[..., :-1, :] / temperature
    shift_mask = attention_mask[..., :-1]

    # Softmax to get probabilities
    p_target = F.softmax(shift_target, dim=-1)  # (batch, seq-1, vocab)
    p_draft = F.softmax(shift_draft, dim=-1)

    # Get draft's top-k token indices
    _, top_indices = torch.topk(p_draft, k=min(top_k, p_draft.size(-1)), dim=-1)
    # top_indices: (batch, seq-1, top_k)

    # Gather probabilities at top-k positions
    p_target_topk = torch.gather(p_target, dim=-1, index=top_indices)
    p_draft_topk = torch.gather(p_draft, dim=-1, index=top_indices)

    # Overlap = sum of min(p_target, p_draft) over top-k tokens per position
    overlap_per_pos = torch.min(p_target_topk, p_draft_topk).sum(dim=-1)
    # overlap_per_pos: (batch, seq-1)

    # Masked mean
    mask_sum = shift_mask.float().sum().clamp(min=1.0)
    overlap_mean = (overlap_per_pos * shift_mask.float()).sum() / mask_sum

    return {
        "overlap_loss": -overlap_mean,  # negate so minimizing = maximizing overlap
        "overlap_value": overlap_mean.detach().item(),
    }
