# tests/test_acceptance_proxy.py
import torch
import pytest
from src.acceptance_proxy_loss import compute_overlap_loss, compute_soft_accept_loss


def test_overlap_loss_identical_logits():
    """When target == draft, overlap should be maximal (loss most negative).

    Uses top_k=90 over vocab=100 so the top-k tokens cover >98% of probability
    mass, making the >0.9 threshold reliable regardless of random seed.
    """
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 100)  # (batch, seq, vocab)
    mask = torch.ones(2, 10)
    result = compute_overlap_loss(
        target_logits=logits.clone(),
        draft_logits=logits.clone(),
        attention_mask=mask,
        top_k=90,  # covers ~99% of mass for vocab=100, so overlap > 0.9
    )
    assert "overlap_loss" in result
    assert "overlap_value" in result
    assert result["overlap_value"] > 0.9  # near-perfect overlap when identical


def test_overlap_loss_different_logits():
    """When target != draft, overlap should be lower."""
    torch.manual_seed(42)
    target_logits = torch.randn(2, 10, 100)
    draft_logits = torch.randn(2, 10, 100)
    mask = torch.ones(2, 10)
    result_diff = compute_overlap_loss(
        target_logits=target_logits,
        draft_logits=draft_logits,
        attention_mask=mask,
        top_k=50,
    )
    result_same = compute_overlap_loss(
        target_logits=target_logits,
        draft_logits=target_logits.clone(),
        attention_mask=mask,
        top_k=50,
    )
    assert result_diff["overlap_value"] < result_same["overlap_value"]


def test_overlap_loss_gradient_flows_through_target():
    """Gradients should flow through target_logits when draft is detached."""
    torch.manual_seed(42)
    target_logits = torch.randn(2, 10, 100, requires_grad=True)
    draft_logits = torch.randn(2, 10, 100)
    mask = torch.ones(2, 10)
    result = compute_overlap_loss(
        target_logits=target_logits,
        draft_logits=draft_logits.detach(),
        attention_mask=mask,
        top_k=50,
    )
    result["overlap_loss"].backward()
    assert target_logits.grad is not None
    assert target_logits.grad.abs().sum() > 0


def test_overlap_loss_gradient_flows_through_draft():
    """Gradients should flow through draft_logits when target is detached."""
    torch.manual_seed(42)
    target_logits = torch.randn(2, 10, 100)
    draft_logits = torch.randn(2, 10, 100, requires_grad=True)
    mask = torch.ones(2, 10)
    result = compute_overlap_loss(
        target_logits=target_logits.detach(),
        draft_logits=draft_logits,
        attention_mask=mask,
        top_k=50,
    )
    result["overlap_loss"].backward()
    assert draft_logits.grad is not None
    assert draft_logits.grad.abs().sum() > 0


def test_overlap_loss_respects_mask():
    """Padding positions should not contribute to loss."""
    torch.manual_seed(42)
    target_logits = torch.randn(1, 5, 100)
    draft_logits = torch.randn(1, 5, 100)
    full_mask = torch.ones(1, 5)
    partial_mask = torch.tensor([[1, 1, 1, 0, 0]])
    result_full = compute_overlap_loss(target_logits, draft_logits, full_mask, top_k=50)
    result_partial = compute_overlap_loss(target_logits, draft_logits, partial_mask, top_k=50)
    assert result_full["overlap_value"] != pytest.approx(result_partial["overlap_value"], abs=1e-4)


def test_soft_accept_loss_identical_logits():
    """When target == draft, acceptance should be ~1.0."""
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 100)
    mask = torch.ones(2, 10)
    result = compute_soft_accept_loss(
        target_logits=logits.clone(),
        draft_logits=logits.clone(),
        attention_mask=mask,
        tau=0.5,
        num_samples=4,
    )
    assert "accept_loss" in result
    assert "accept_value" in result
    assert result["accept_value"] > 0.8  # high acceptance for identical distributions


def test_soft_accept_loss_gradient_flows():
    """Gradients flow through draft when target is detached."""
    torch.manual_seed(42)
    target_logits = torch.randn(2, 10, 100)
    draft_logits = torch.randn(2, 10, 100, requires_grad=True)
    mask = torch.ones(2, 10)
    result = compute_soft_accept_loss(
        target_logits=target_logits.detach(),
        draft_logits=draft_logits,
        attention_mask=mask,
        tau=0.5,
        num_samples=4,
    )
    result["accept_loss"].backward()
    assert draft_logits.grad is not None
    assert draft_logits.grad.abs().sum() > 0
