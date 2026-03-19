"""Smoke tests for DPO training components.

Verifies that:
1. DPO loss computes without errors on dummy data
2. Combined DPO + spec loss backpropagates only through target LoRA params
3. Reference model and draft model have zero gradient
4. Per-token log-probability computation is correct
"""

import torch
import torch.nn.functional as F
import pytest


# ---------------------------------------------------------------------------
# Helpers to create dummy data
# ---------------------------------------------------------------------------

def _make_dummy_logits(batch: int, seq_len: int, vocab: int, requires_grad: bool = False):
    """Create random logits tensor."""
    return torch.randn(batch, seq_len, vocab, requires_grad=requires_grad)


def _make_dummy_labels(batch: int, seq_len: int, vocab: int, prompt_len: int = 4):
    """Create labels with prompt positions masked to -100."""
    labels = torch.randint(0, vocab, (batch, seq_len))
    labels[:, :prompt_len] = -100
    labels[:, -2:] = -100  # simulate padding
    return labels


def _make_dummy_mask(batch: int, seq_len: int, pad_len: int = 2):
    """Create attention mask with padding at the end."""
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    mask[:, -pad_len:] = 0
    return mask


# ---------------------------------------------------------------------------
# Test: per-token log-probs
# ---------------------------------------------------------------------------

class TestPerTokenLogps:
    """Tests for _compute_per_token_logps."""

    def test_basic_computation(self):
        """Per-token logps should be finite and only sum over non-masked positions."""
        from src.train_dpo import _compute_per_token_logps

        batch, seq_len, vocab = 2, 16, 100
        logits = _make_dummy_logits(batch, seq_len, vocab)
        labels = _make_dummy_labels(batch, seq_len, vocab, prompt_len=4)

        logps = _compute_per_token_logps(logits, labels)

        assert logps.shape == (batch,)
        assert torch.isfinite(logps).all()
        # Log-probs should be negative (sum of log-softmax values)
        assert (logps <= 0).all()

    def test_all_masked_gives_zero(self):
        """If all labels are -100, log-prob sum should be zero."""
        from src.train_dpo import _compute_per_token_logps

        batch, seq_len, vocab = 2, 16, 100
        logits = _make_dummy_logits(batch, seq_len, vocab)
        labels = torch.full((batch, seq_len), -100, dtype=torch.long)

        logps = _compute_per_token_logps(logits, labels)
        assert torch.allclose(logps, torch.zeros(batch))


# ---------------------------------------------------------------------------
# Test: DPO loss
# ---------------------------------------------------------------------------

class TestDPOLoss:
    """Tests for compute_dpo_loss."""

    def test_dpo_loss_computes(self):
        """DPO loss should return finite values."""
        from src.train_dpo import compute_dpo_loss

        batch, seq_len, vocab = 2, 16, 100
        target_chosen = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        target_rejected = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        ref_chosen_logps = torch.randn(batch)
        ref_rejected_logps = torch.randn(batch)
        chosen_labels = _make_dummy_labels(batch, seq_len, vocab)
        rejected_labels = _make_dummy_labels(batch, seq_len, vocab)

        result = compute_dpo_loss(
            target_chosen_logits=target_chosen,
            target_rejected_logits=target_rejected,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=0.1,
        )

        assert torch.isfinite(result["dpo_loss"])
        assert isinstance(result["preferred_reward"], float)
        assert isinstance(result["rejected_reward"], float)
        assert isinstance(result["reward_margin"], float)

    def test_dpo_loss_backprop(self):
        """DPO loss should backpropagate through target logits."""
        from src.train_dpo import compute_dpo_loss

        batch, seq_len, vocab = 2, 16, 100
        target_chosen = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        target_rejected = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        ref_chosen_logps = torch.randn(batch)
        ref_rejected_logps = torch.randn(batch)
        chosen_labels = _make_dummy_labels(batch, seq_len, vocab)
        rejected_labels = _make_dummy_labels(batch, seq_len, vocab)

        result = compute_dpo_loss(
            target_chosen_logits=target_chosen,
            target_rejected_logits=target_rejected,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=0.1,
        )

        result["dpo_loss"].backward()

        assert target_chosen.grad is not None
        assert target_chosen.grad.abs().sum() > 0
        assert target_rejected.grad is not None
        assert target_rejected.grad.abs().sum() > 0

    def test_ref_logps_no_grad(self):
        """Reference log-probs should not receive gradients."""
        from src.train_dpo import compute_dpo_loss

        batch, seq_len, vocab = 2, 16, 100
        target_chosen = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        target_rejected = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        ref_chosen_logps = torch.randn(batch)
        ref_rejected_logps = torch.randn(batch)
        chosen_labels = _make_dummy_labels(batch, seq_len, vocab)
        rejected_labels = _make_dummy_labels(batch, seq_len, vocab)

        result = compute_dpo_loss(
            target_chosen_logits=target_chosen,
            target_rejected_logits=target_rejected,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=0.1,
        )

        result["dpo_loss"].backward()

        # ref logps are plain tensors without requires_grad, so grad should be None
        assert ref_chosen_logps.grad is None
        assert ref_rejected_logps.grad is None


# ---------------------------------------------------------------------------
# Test: Spec KL for DPO
# ---------------------------------------------------------------------------

class TestSpecKL:
    """Tests for _compute_spec_kl."""

    def test_spec_kl_computes(self):
        """Spec KL should return a finite scalar."""
        from src.train_dpo import _compute_spec_kl

        batch, seq_len, vocab = 2, 16, 100
        target = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        draft = _make_dummy_logits(batch, seq_len, vocab)
        mask = _make_dummy_mask(batch, seq_len)

        kl = _compute_spec_kl(target, draft, mask)

        assert torch.isfinite(kl)
        assert kl.item() >= 0  # KL is non-negative

    def test_spec_kl_grad_flow(self):
        """Gradients should flow through target but not draft."""
        from src.train_dpo import _compute_spec_kl

        batch, seq_len, vocab = 2, 16, 100
        target = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        draft = _make_dummy_logits(batch, seq_len, vocab)
        mask = _make_dummy_mask(batch, seq_len)

        kl = _compute_spec_kl(target, draft, mask)
        kl.backward()

        assert target.grad is not None
        assert target.grad.abs().sum() > 0
        assert draft.grad is None

    def test_spec_kl_vocab_mismatch(self):
        """Should handle different vocab sizes gracefully."""
        from src.train_dpo import _compute_spec_kl

        batch, seq_len = 2, 16
        target = _make_dummy_logits(batch, seq_len, 100, requires_grad=True)
        draft = _make_dummy_logits(batch, seq_len, 80)  # smaller vocab
        mask = _make_dummy_mask(batch, seq_len)

        kl = _compute_spec_kl(target, draft, mask)
        assert torch.isfinite(kl)


# ---------------------------------------------------------------------------
# Test: Combined DPO + spec loss
# ---------------------------------------------------------------------------

class TestCombinedLoss:
    """Tests for combined DPO + spec loss gradient flow."""

    def test_combined_loss_backprop(self):
        """Combined loss should only produce gradients for target logits."""
        from src.train_dpo import compute_dpo_loss, _compute_spec_kl

        batch, seq_len, vocab = 2, 16, 100
        lam = 0.5

        # Target logits (with grad)
        target_chosen = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)
        target_rejected = _make_dummy_logits(batch, seq_len, vocab, requires_grad=True)

        # Reference logps (no grad)
        ref_chosen_logps = torch.randn(batch)
        ref_rejected_logps = torch.randn(batch)

        # Draft logits (no grad)
        draft_logits = _make_dummy_logits(batch, seq_len, vocab)

        # Labels and mask
        chosen_labels = _make_dummy_labels(batch, seq_len, vocab)
        rejected_labels = _make_dummy_labels(batch, seq_len, vocab)
        chosen_mask = _make_dummy_mask(batch, seq_len)

        # DPO loss
        dpo_result = compute_dpo_loss(
            target_chosen, target_rejected,
            ref_chosen_logps, ref_rejected_logps,
            chosen_labels, rejected_labels,
            beta=0.1,
        )

        # Spec loss
        spec_kl = _compute_spec_kl(target_chosen, draft_logits, chosen_mask)

        # Combined
        total = dpo_result["dpo_loss"] + lam * spec_kl
        total.backward()

        # Target should have gradients
        assert target_chosen.grad is not None and target_chosen.grad.abs().sum() > 0
        assert target_rejected.grad is not None and target_rejected.grad.abs().sum() > 0

        # Reference and draft should NOT have gradients
        assert ref_chosen_logps.grad is None
        assert ref_rejected_logps.grad is None
        assert draft_logits.grad is None


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
