"""Speculator-aware loss functions for fine-tuning with draft-model alignment.

This module implements the core loss computation for speculator-aware fine-tuning.
The combined loss is: L_total = L_task + lambda * L_spec

Supported speculator loss types:
    - kl: Forward KL divergence KL(target || draft)
    - reverse_kl: Reverse KL divergence KL(draft || target)
    - js: Jensen-Shannon divergence
    - tv: Total Variation distance
    - token_match: Fraction of positions where top-1 tokens differ
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPSILON = 1e-10


def _compute_task_loss(
    target_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute causal LM cross-entropy loss with label shifting.

    Args:
        target_logits: (batch, seq_len, vocab_size) logits from the target model.
        labels: (batch, seq_len) ground truth token IDs. Padding uses -100.

    Returns:
        Scalar cross-entropy loss with gradients.
    """
    # Shift: logits[..., :-1, :] predict labels[..., 1:]
    shift_logits = target_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def _get_probs(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert logits to probabilities, optionally restricting to top-k.

    Args:
        logits: (batch, seq_len, vocab_size) raw logits.
        temperature: Softmax temperature.
        top_k: If set, restrict to top-k logits and renormalize.

    Returns:
        Tuple of (probs, log_probs, top_k_indices or None).
        If top_k is set, probs and log_probs have shape (batch, seq_len, top_k).
        Otherwise shape is (batch, seq_len, vocab_size).
    """
    scaled = logits / temperature

    if top_k is not None and top_k < logits.size(-1):
        top_k_values, top_k_indices = torch.topk(scaled, top_k, dim=-1)
        probs = F.softmax(top_k_values, dim=-1)
        log_probs = F.log_softmax(top_k_values, dim=-1)
        return probs, log_probs, top_k_indices
    else:
        probs = F.softmax(scaled, dim=-1)
        log_probs = F.log_softmax(scaled, dim=-1)
        return probs, log_probs, None


def _gather_to_topk(
    logits: torch.Tensor,
    indices: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logits at top-k indices and compute renormalized probs/log_probs.

    Args:
        logits: (batch, seq_len, vocab_size) raw logits.
        indices: (batch, seq_len, top_k) indices to gather.
        temperature: Softmax temperature.

    Returns:
        Tuple of (probs, log_probs), each (batch, seq_len, top_k).
    """
    scaled = logits / temperature
    gathered = torch.gather(scaled, dim=-1, index=indices)
    probs = F.softmax(gathered, dim=-1)
    log_probs = F.log_softmax(gathered, dim=-1)
    return probs, log_probs


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean of values over positions where mask == 1.

    Args:
        values: (batch, seq_len) per-position values.
        mask: (batch, seq_len) binary mask.

    Returns:
        Scalar mean.
    """
    mask = mask.float()
    total = (values * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count


def _kl_divergence(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Compute forward KL divergence: KL(target || draft).

    Args:
        target_logits: (batch, seq_len, vocab_size) with gradients.
        draft_logits: (batch, seq_len, vocab_size) detached.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.
        top_k: If set, restrict computation to top-k logits.

    Returns:
        Scalar KL divergence, masked and averaged.
    """
    target_probs, target_log_probs, topk_indices = _get_probs(
        target_logits, temperature, top_k
    )

    if topk_indices is not None:
        draft_probs, draft_log_probs = _gather_to_topk(
            draft_logits, topk_indices, temperature
        )
    else:
        draft_probs, draft_log_probs, _ = _get_probs(
            draft_logits, temperature, None
        )

    # KL(target || draft) = sum_x p_target(x) * (log p_target(x) - log p_draft(x))
    # Clamp draft log probs to avoid -inf
    draft_log_probs_safe = torch.clamp(draft_log_probs, min=math.log(EPSILON))
    kl_per_pos = (target_probs * (target_log_probs - draft_log_probs_safe)).sum(dim=-1)

    # Use shifted mask (matching shifted logits for task loss alignment)
    mask = attention_mask[..., :-1]
    kl_per_pos = kl_per_pos[..., :-1]

    return _masked_mean(kl_per_pos, mask)


def _reverse_kl_divergence(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Compute reverse KL divergence: KL(draft || target).

    This is mode-seeking: penalizes the target for NOT covering modes
    that the draft covers.

    Args:
        target_logits: (batch, seq_len, vocab_size) with gradients.
        draft_logits: (batch, seq_len, vocab_size) detached.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.
        top_k: If set, restrict computation to top-k logits.

    Returns:
        Scalar reverse KL divergence, masked and averaged.
    """
    target_probs, target_log_probs, topk_indices = _get_probs(
        target_logits, temperature, top_k
    )

    if topk_indices is not None:
        draft_probs, draft_log_probs = _gather_to_topk(
            draft_logits, topk_indices, temperature
        )
    else:
        draft_probs, draft_log_probs, _ = _get_probs(
            draft_logits, temperature, None
        )

    # KL(draft || target) = sum_x p_draft(x) * (log p_draft(x) - log p_target(x))
    target_log_probs_safe = torch.clamp(target_log_probs, min=math.log(EPSILON))
    kl_per_pos = (draft_probs * (draft_log_probs - target_log_probs_safe)).sum(dim=-1)

    mask = attention_mask[..., :-1]
    kl_per_pos = kl_per_pos[..., :-1]

    return _masked_mean(kl_per_pos, mask)


def _js_divergence(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence: JS(target, draft).

    JS = 0.5 * KL(target || m) + 0.5 * KL(draft || m), where m = 0.5 * (p_target + p_draft).
    JS is symmetric and bounded in [0, ln(2)].

    Args:
        target_logits: (batch, seq_len, vocab_size) with gradients.
        draft_logits: (batch, seq_len, vocab_size) detached.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.
        top_k: If set, restrict computation to top-k logits.

    Returns:
        Scalar JS divergence, masked and averaged.
    """
    target_probs, target_log_probs, topk_indices = _get_probs(
        target_logits, temperature, top_k
    )

    if topk_indices is not None:
        draft_probs, draft_log_probs = _gather_to_topk(
            draft_logits, topk_indices, temperature
        )
    else:
        draft_probs, draft_log_probs, _ = _get_probs(
            draft_logits, temperature, None
        )

    # m = 0.5 * (p_target + p_draft)
    m = 0.5 * (target_probs + draft_probs)
    log_m = torch.log(m + EPSILON)

    # KL(target || m)
    kl_target_m = (target_probs * (target_log_probs - log_m)).sum(dim=-1)
    # KL(draft || m)
    kl_draft_m = (draft_probs * (draft_log_probs - log_m)).sum(dim=-1)

    js_per_pos = 0.5 * kl_target_m + 0.5 * kl_draft_m

    mask = attention_mask[..., :-1]
    js_per_pos = js_per_pos[..., :-1]

    return _masked_mean(js_per_pos, mask)


def _tv_distance(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Compute Total Variation distance: TV = 0.5 * sum(|p_target - p_draft|).

    TV is directly related to acceptance rate in speculative decoding.

    Args:
        target_logits: (batch, seq_len, vocab_size) with gradients.
        draft_logits: (batch, seq_len, vocab_size) detached.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.
        top_k: If set, restrict computation to top-k logits.

    Returns:
        Scalar TV distance, masked and averaged.
    """
    target_probs, _, topk_indices = _get_probs(target_logits, temperature, top_k)

    if topk_indices is not None:
        draft_probs, _ = _gather_to_topk(draft_logits, topk_indices, temperature)
    else:
        draft_probs, _, _ = _get_probs(draft_logits, temperature, None)

    tv_per_pos = 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)

    mask = attention_mask[..., :-1]
    tv_per_pos = tv_per_pos[..., :-1]

    return _masked_mean(tv_per_pos, mask)


def _token_match_loss(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Compute token match loss: fraction of positions where top-1 tokens differ.

    Since argmax is not differentiable, we use a soft approximation:
    compute the probability that the target model assigns to the draft model's
    top-1 token, and use 1 - mean(p_target(argmax_draft)) as a differentiable
    proxy. This provides gradients that push the target's distribution toward
    assigning more mass to the draft's top token.

    Args:
        target_logits: (batch, seq_len, vocab_size) with gradients.
        draft_logits: (batch, seq_len, vocab_size) detached.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.
        top_k: Ignored for token match (operates on argmax only).

    Returns:
        Scalar token match loss, masked and averaged.
    """
    target_probs = F.softmax(target_logits / temperature, dim=-1)

    # Draft's top-1 token at each position
    draft_top1 = draft_logits.argmax(dim=-1)  # (batch, seq_len)

    # Probability target assigns to draft's top-1 token
    # gather along vocab dimension
    draft_top1_unsqueezed = draft_top1.unsqueeze(-1)  # (batch, seq_len, 1)
    p_target_at_draft_top1 = torch.gather(
        target_probs, dim=-1, index=draft_top1_unsqueezed
    ).squeeze(-1)  # (batch, seq_len)

    # Loss = 1 - p_target(argmax_draft): higher when target disagrees with draft
    loss_per_pos = 1.0 - p_target_at_draft_top1

    mask = attention_mask[..., :-1]
    loss_per_pos = loss_per_pos[..., :-1]

    return _masked_mean(loss_per_pos, mask)


def _compute_acceptance_proxy(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> float:
    """Estimate speculative decoding acceptance rate during training.

    Computes min(1, p_target(x_draft) / p_draft(x_draft)) averaged over
    non-padding positions, where x_draft = argmax(draft_logits).

    This is an approximation of the true acceptance probability in speculative
    decoding. It is detached and used for monitoring only.

    Args:
        target_logits: (batch, seq_len, vocab_size) from target model.
        draft_logits: (batch, seq_len, vocab_size) from draft model.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.

    Returns:
        Scalar float: estimated mean acceptance rate.
    """
    with torch.no_grad():
        target_probs = F.softmax(target_logits / temperature, dim=-1)
        draft_probs = F.softmax(draft_logits / temperature, dim=-1)

        # Draft's greedy token
        x_draft = draft_logits.argmax(dim=-1)  # (batch, seq_len)
        x_draft_unsqueezed = x_draft.unsqueeze(-1)  # (batch, seq_len, 1)

        # p_target(x_draft) and p_draft(x_draft)
        p_target = torch.gather(target_probs, dim=-1, index=x_draft_unsqueezed).squeeze(-1)
        p_draft = torch.gather(draft_probs, dim=-1, index=x_draft_unsqueezed).squeeze(-1)

        # Acceptance probability: min(1, p_target / p_draft)
        ratio = p_target / (p_draft + EPSILON)
        acceptance = torch.clamp(ratio, max=1.0)

        # Average over non-padding positions (use shifted mask for consistency)
        mask = attention_mask[..., :-1].float()
        acceptance = acceptance[..., :-1]

        total = (acceptance * mask).sum()
        count = mask.sum().clamp(min=1.0)

        return (total / count).item()


# Registry of loss functions
_SPEC_LOSS_FNS = {
    "kl": _kl_divergence,
    "reverse_kl": _reverse_kl_divergence,
    "js": _js_divergence,
    "tv": _tv_distance,
    "token_match": _token_match_loss,
}


def compute_spec_loss(
    target_logits: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    lam: float = 0.1,
    loss_type: str = "kl",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> dict[str, torch.Tensor | float]:
    """Compute the combined speculator-aware fine-tuning loss.

    L_total = L_task + lam * L_spec

    Where L_task is the standard causal LM cross-entropy and L_spec is the
    speculator alignment loss that penalizes distribution drift from the draft model.

    Args:
        target_logits: (batch, seq_len, vocab_size) logits from the target model.
            Must have requires_grad=True for backpropagation.
        draft_logits: (batch, seq_len, vocab_size) logits from the frozen draft model.
            Will be detached to ensure no gradient flow.
        labels: (batch, seq_len) ground truth token IDs. Use -100 for padding.
        attention_mask: (batch, seq_len) binary mask, 1 for real tokens, 0 for padding.
        lam: Regularization strength (lambda). 0.0 disables spec loss.
        loss_type: One of "kl", "reverse_kl", "js", "tv", "token_match".
        temperature: Softmax temperature for divergence computation.
        top_k: If set, restrict spec loss to top-k logits for efficiency.

    Returns:
        Dict with keys:
            - loss: Combined loss tensor with gradients for backprop.
            - task_loss: Cross-entropy value (detached float) for logging.
            - spec_loss: Speculator loss value (detached float) for logging.
            - acceptance_proxy: Estimated acceptance rate (detached float) for monitoring.

    Raises:
        ValueError: If loss_type is not recognized.
    """
    if loss_type not in _SPEC_LOSS_FNS:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Must be one of: {list(_SPEC_LOSS_FNS.keys())}"
        )

    # Task loss: standard causal LM cross-entropy
    task_loss = _compute_task_loss(target_logits, labels)

    # Handle draft_logits=None (standard fine-tuning, lam=0)
    if draft_logits is None or lam == 0.0:
        return {
            "loss": task_loss,
            "task_loss": task_loss.detach().item(),
            "spec_loss": 0.0,
            "acceptance_proxy": 0.0,
        }

    # Ensure draft logits are detached and on the same device as target
    draft_logits = draft_logits.detach().to(target_logits.device)

    # Speculator loss
    spec_loss_fn = _SPEC_LOSS_FNS[loss_type]
    spec_loss = spec_loss_fn(
        target_logits, draft_logits, attention_mask, temperature, top_k
    )

    # Combined loss
    combined_loss = task_loss + lam * spec_loss

    # Acceptance proxy (monitoring only, no gradients)
    acceptance_proxy = _compute_acceptance_proxy(
        target_logits, draft_logits, attention_mask, temperature
    )

    return {
        "loss": combined_loss,
        "task_loss": task_loss.detach().item(),
        "spec_loss": spec_loss.detach().item(),
        "acceptance_proxy": acceptance_proxy,
    }


if __name__ == "__main__":
    """Test all loss types with random tensors. Verify gradient flow."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    vocab_size = 100

    # Target logits: requires grad (simulates trainable target model)
    target_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    # Draft logits: no grad (simulates frozen draft model)
    draft_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Random labels (valid token IDs, with some padding marked as -100)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, -3:] = -100  # last 3 positions are padding

    # Attention mask
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[:, -3:] = 0  # mask padding

    loss_types = ["kl", "reverse_kl", "js", "tv", "token_match"]

    print("=" * 70)
    print("Speculator-Aware Loss Function Tests")
    print("=" * 70)
    print(f"Shape: batch={batch_size}, seq_len={seq_len}, vocab={vocab_size}")
    print()

    all_passed = True

    for lt in loss_types:
        # Fresh target logits for each test (gradients get consumed by backward)
        t_logits = target_logits.detach().clone().requires_grad_(True)
        d_logits = draft_logits.detach().clone()

        result = compute_spec_loss(
            target_logits=t_logits,
            draft_logits=d_logits,
            labels=labels,
            attention_mask=attention_mask,
            lam=0.1,
            loss_type=lt,
            temperature=1.0,
            top_k=None,
        )

        # Check finiteness
        loss_finite = torch.isfinite(result["loss"]).item()
        task_finite = not (result["task_loss"] != result["task_loss"])  # NaN check
        spec_finite = not (result["spec_loss"] != result["spec_loss"])

        # Backward pass
        result["loss"].backward()

        # Check gradients
        target_has_grad = t_logits.grad is not None and t_logits.grad.abs().sum() > 0
        draft_no_grad = d_logits.grad is None

        passed = all([loss_finite, task_finite, spec_finite, target_has_grad, draft_no_grad])
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"[{status}] loss_type={lt:12s} | "
              f"loss={result['loss'].item():.4f} | "
              f"task_loss={result['task_loss']:.4f} | "
              f"spec_loss={result['spec_loss']:.4f} | "
              f"acceptance_proxy={result['acceptance_proxy']:.4f} | "
              f"target_grad={'yes' if target_has_grad else 'NO'} | "
              f"draft_grad={'none' if draft_no_grad else 'LEAKED'}")

    # Test with top_k
    print()
    print("--- Testing with top_k=20 ---")
    for lt in ["kl", "js"]:
        t_logits = target_logits.detach().clone().requires_grad_(True)
        d_logits = draft_logits.detach().clone()

        result = compute_spec_loss(
            target_logits=t_logits,
            draft_logits=d_logits,
            labels=labels,
            attention_mask=attention_mask,
            lam=0.1,
            loss_type=lt,
            temperature=1.0,
            top_k=20,
        )
        result["loss"].backward()
        target_has_grad = t_logits.grad is not None and t_logits.grad.abs().sum() > 0
        passed = torch.isfinite(result["loss"]).item() and target_has_grad
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] loss_type={lt:12s} top_k=20 | "
              f"loss={result['loss'].item():.4f} | "
              f"spec_loss={result['spec_loss']:.4f}")

    # Test with lam=0.0 (spec loss should be zero)
    print()
    print("--- Testing with lam=0.0 (baseline, no spec loss) ---")
    t_logits = target_logits.detach().clone().requires_grad_(True)
    d_logits = draft_logits.detach().clone()
    result = compute_spec_loss(
        target_logits=t_logits,
        draft_logits=d_logits,
        labels=labels,
        attention_mask=attention_mask,
        lam=0.0,
        loss_type="kl",
    )
    result["loss"].backward()
    lam0_ok = result["spec_loss"] == 0.0 and t_logits.grad is not None
    status = "PASS" if lam0_ok else "FAIL"
    if not lam0_ok:
        all_passed = False
    print(f"[{status}] lam=0.0 | "
          f"loss={result['loss'].item():.4f} | "
          f"spec_loss={result['spec_loss']:.4f} (should be 0.0)")

    print()
    print("=" * 70)
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED. Check output above.")
    print("=" * 70)
