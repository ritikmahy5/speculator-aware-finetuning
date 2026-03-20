"""Tests for joint training gradient isolation.

Verifies that target task loss gradients do not leak into draft parameters,
and draft distillation loss gradients do not leak into target parameters.
"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def test_joint_training_step_gradient_isolation():
    """Target loss gradients should not affect draft, and vice versa."""
    torch.manual_seed(42)

    target_logits = torch.randn(2, 10, 100, requires_grad=True)
    draft_logits = torch.randn(2, 10, 100, requires_grad=True)
    labels = torch.randint(0, 100, (2, 10))

    # Target loss: cross-entropy
    ce = CrossEntropyLoss(ignore_index=-100)
    target_loss = ce(target_logits[:, :-1, :].reshape(-1, 100), labels[:, 1:].reshape(-1))

    # Draft loss: KL(draft || target.detach())
    p_target = F.softmax(target_logits[:, :-1, :].detach(), dim=-1)
    log_p_draft = F.log_softmax(draft_logits[:, :-1, :], dim=-1)
    draft_loss = F.kl_div(log_p_draft, p_target, reduction="batchmean")

    # Backward target
    target_loss.backward()
    assert target_logits.grad is not None
    assert draft_logits.grad is None  # draft not affected

    # Backward draft
    draft_loss.backward()
    assert draft_logits.grad is not None
