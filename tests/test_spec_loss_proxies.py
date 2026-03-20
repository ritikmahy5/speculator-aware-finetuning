"""Tests for overlap and soft_accept proxy losses integrated into spec_loss."""

import torch
from src.spec_loss import compute_spec_loss


def test_spec_loss_overlap_type():
    """compute_spec_loss should accept loss_type='overlap'."""
    torch.manual_seed(42)
    target_logits = torch.randn(2, 10, 100, requires_grad=True)
    draft_logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10)
    result = compute_spec_loss(
        target_logits=target_logits,
        draft_logits=draft_logits,
        labels=labels,
        attention_mask=mask,
        lam=0.5,
        loss_type="overlap",
    )
    assert "loss" in result
    assert "task_loss" in result
    assert "spec_loss" in result
    result["loss"].backward()
    assert target_logits.grad is not None


def test_spec_loss_soft_accept_type():
    """compute_spec_loss should accept loss_type='soft_accept'."""
    torch.manual_seed(42)
    target_logits = torch.randn(2, 10, 100, requires_grad=True)
    draft_logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10)
    result = compute_spec_loss(
        target_logits=target_logits,
        draft_logits=draft_logits,
        labels=labels,
        attention_mask=mask,
        lam=0.5,
        loss_type="soft_accept",
    )
    assert "loss" in result
    result["loss"].backward()
    assert target_logits.grad is not None
