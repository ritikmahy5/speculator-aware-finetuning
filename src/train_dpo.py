"""DPO training script with optional speculator-aware KL regularization.

Implements Direct Preference Optimization with an optional KL(target || draft)
regularization term that preserves speculative decoding acceptance rates during
alignment training.

    L_total = L_DPO(preferred, rejected) + lambda * KL(p_target || p_draft)

Three models are loaded simultaneously:
    1. Target model (with LoRA, cuda:0) — the model being trained
    2. Reference model (frozen, cuda:0, 8-bit) — pre-DPO target for pi_ref
    3. Draft model (frozen, cuda:1) — speculator for KL regularization

Usage:
    python -m src.train_dpo --config configs/exp_dpo_baseline.yaml
    python -m src.train_dpo --config configs/exp_dpo_specaware.yaml
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.spec_loss import _compute_acceptance_proxy, EPSILON

logger = logging.getLogger("specaware")


# ---------------------------------------------------------------------------
# Scheduler (reused from train.py)
# ---------------------------------------------------------------------------

def _create_scheduler(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a linear-warmup then cosine-decay learning-rate scheduler."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# DPO loss computation
# ---------------------------------------------------------------------------

def _compute_per_token_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities for completion tokens only.

    Args:
        logits: (batch, seq_len, vocab_size) model output logits.
        labels: (batch, seq_len) token IDs with -100 for prompt/padding.

    Returns:
        (batch,) sum of log-probabilities over completion tokens per sample.
    """
    # Shift: logits[:-1] predict labels[1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Log-softmax over vocab
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probs at the label positions
    # Replace -100 with 0 for gathering (will be masked out)
    gather_labels = shift_labels.clone()
    mask = gather_labels != -100
    gather_labels[~mask] = 0

    per_token_logps = torch.gather(
        log_probs, dim=-1, index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Zero out prompt/padding positions
    per_token_logps = per_token_logps * mask.float()

    # Sum per sample
    return per_token_logps.sum(dim=-1)


def compute_dpo_loss(
    target_chosen_logits: torch.Tensor,
    target_rejected_logits: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, torch.Tensor | float]:
    """Compute the DPO loss.

    L_DPO = -log(sigma(beta * (log(pi_theta(y_w|x)/pi_ref(y_w|x))
                              - log(pi_theta(y_l|x)/pi_ref(y_l|x)))))

    Args:
        target_chosen_logits: (batch, seq_len, vocab) from target on chosen.
        target_rejected_logits: (batch, seq_len, vocab) from target on rejected.
        ref_chosen_logps: (batch,) pre-computed log-probs from reference on chosen.
        ref_rejected_logps: (batch,) pre-computed log-probs from reference on rejected.
        chosen_labels: (batch, seq_len) labels for chosen (prompt masked to -100).
        rejected_labels: (batch, seq_len) labels for rejected (prompt masked to -100).
        beta: DPO temperature parameter.

    Returns:
        Dict with: dpo_loss (tensor), preferred_reward, rejected_reward,
        reward_margin (all detached floats).
    """
    # Current policy log-probs
    target_chosen_logps = _compute_per_token_logps(target_chosen_logits, chosen_labels)
    target_rejected_logps = _compute_per_token_logps(target_rejected_logits, rejected_labels)

    # Log-ratios: log(pi_theta / pi_ref)
    chosen_log_ratio = target_chosen_logps - ref_chosen_logps
    rejected_log_ratio = target_rejected_logps - ref_rejected_logps

    # DPO loss
    logits_diff = beta * (chosen_log_ratio - rejected_log_ratio)
    dpo_loss = -F.logsigmoid(logits_diff).mean()

    # Rewards for logging (implicit rewards under DPO)
    with torch.no_grad():
        preferred_reward = beta * chosen_log_ratio.mean().item()
        rejected_reward = beta * rejected_log_ratio.mean().item()
        reward_margin = preferred_reward - rejected_reward

    return {
        "dpo_loss": dpo_loss,
        "preferred_reward": preferred_reward,
        "rejected_reward": rejected_reward,
        "reward_margin": reward_margin,
    }


# ---------------------------------------------------------------------------
# Spec loss for DPO (operates on chosen sequence)
# ---------------------------------------------------------------------------

def _compute_spec_kl(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute forward KL(target || draft) for DPO spec regularization.

    Simplified version of the full spec_loss module — just forward KL, which
    is the best loss at high lambda per our EXP-6 results.

    Args:
        target_logits: (batch, seq_len, vocab_size) with gradients.
        draft_logits: (batch, seq_len, vocab_size) detached.
        attention_mask: (batch, seq_len) binary mask.
        temperature: Softmax temperature.

    Returns:
        Scalar KL divergence, masked and averaged.
    """
    draft_logits = draft_logits.detach().to(target_logits.device)

    # Align vocab sizes
    target_vocab = target_logits.size(-1)
    draft_vocab = draft_logits.size(-1)
    if target_vocab != draft_vocab:
        min_vocab = min(target_vocab, draft_vocab)
        target_logits = target_logits[..., :min_vocab]
        draft_logits = draft_logits[..., :min_vocab]

    target_probs = F.softmax(target_logits / temperature, dim=-1)
    target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)
    draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)
    draft_log_probs_safe = torch.clamp(draft_log_probs, min=math.log(EPSILON))

    kl_per_pos = (target_probs * (target_log_probs - draft_log_probs_safe)).sum(dim=-1)

    # Use shifted mask for consistency with causal LM convention
    mask = attention_mask[:, :-1].float()
    kl_per_pos = kl_per_pos[:, :-1]

    total = (kl_per_pos * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _log_param_counts(model: Any) -> None:
    """Log total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameters — total: %s, trainable: %s (%.2f%%)",
        f"{total:,}", f"{trainable:,}",
        100.0 * trainable / total if total > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_dpo(
    config: dict,
    target_device: torch.device,
    draft_device: torch.device,
) -> None:
    """Run DPO training with optional speculator-aware KL regularization.

    Args:
        config: Fully-merged experiment config dictionary.
        target_device: Device for target and reference models.
        draft_device: Device for the draft model.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data_dpo import create_dpo_dataloader, load_dpo_dataset
    from src.utils import (
        save_checkpoint,
        save_config,
        save_metrics,
        set_seed,
        setup_logging,
    )

    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    training_cfg: dict = config.get("training", {})
    lora_cfg: dict = config.get("lora", {})
    spec_cfg: dict = config.get("spec_loss", {})
    dpo_cfg: dict = config.get("dpo", {})
    logging_cfg: dict = config.get("logging", {})

    seed: int = training_cfg.get("seed", 42)
    set_seed(seed)

    output_dir: str = config.get("output_dir", "results/exp_dpo/default")
    log = setup_logging(output_dir)
    save_config(config, output_dir)

    log.info("Experiment: %s", config.get("experiment_id", "N/A"))
    log.info("Target device: %s | Draft device: %s", target_device, draft_device)

    lam: float = float(spec_cfg.get("lam", 0.0))
    loss_type: str = spec_cfg.get("loss_type", "kl")
    temperature: float = float(spec_cfg.get("temperature", 1.0))
    beta: float = float(dpo_cfg.get("beta", 0.1))
    use_bf16: bool = training_cfg.get("bf16", True)
    reference_free: bool = dpo_cfg.get("reference_free", False)

    log.info("DPO beta: %.2f | Spec lambda: %.4f | Loss type: %s", beta, lam, loss_type)
    log.info("Reference-free mode: %s", reference_free)

    # ---------------------------------------------------------------
    # 2. Load tokenizer
    # ---------------------------------------------------------------
    target_name: str = config["models"]["target"]
    draft_name: str = config["models"]["draft"]

    log.info("Loading tokenizer from %s", target_name)
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("Set pad_token to eos_token (%s)", tokenizer.eos_token)

    # ---------------------------------------------------------------
    # 3. Load target model + apply LoRA
    # ---------------------------------------------------------------
    log.info("Loading target model: %s", target_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=torch.bfloat16,
    )
    target_model.to(target_device)

    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    target_model = get_peft_model(target_model, lora_config)
    target_model.gradient_checkpointing_enable()
    _log_param_counts(target_model)

    # ---------------------------------------------------------------
    # 4. Load reference model (frozen, 8-bit on same GPU as target)
    # ---------------------------------------------------------------
    if not reference_free:
        log.info("Loading reference model: %s (frozen, 8-bit)", target_name)
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            target_name,
            quantization_config=bnb_config,
            device_map={"": target_device},
            torch_dtype=torch.bfloat16,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        log.info("Reference model loaded and frozen on %s (8-bit)", target_device)
    else:
        ref_model = None
        log.info(
            "Reference-free mode: will use target model with disabled adapters "
            "for reference log-probs."
        )

    # ---------------------------------------------------------------
    # 5. Load draft model (frozen, only if lambda > 0)
    # ---------------------------------------------------------------
    draft_model: AutoModelForCausalLM | None = None

    if lam > 0:
        log.info("Loading draft model: %s (frozen)", draft_name)
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_name,
            torch_dtype=torch.bfloat16,
        )
        draft_model.to(draft_device)
        draft_model.eval()
        for param in draft_model.parameters():
            param.requires_grad = False
        log.info("Draft model loaded and frozen on %s", draft_device)
    else:
        log.info("Lambda=0 — skipping draft model (standard DPO).")

    # ---------------------------------------------------------------
    # 6. Load DPO preference data
    # ---------------------------------------------------------------
    dataset_name: str = dpo_cfg.get("dataset", "HuggingFaceH4/ultrafeedback_binarized")
    max_samples: int = training_cfg.get("max_samples", 10000)
    max_seq_len: int = training_cfg.get("max_seq_len", 1024)
    batch_size: int = training_cfg.get("batch_size", 2)

    log.info("Loading DPO dataset: %s (max_samples=%d)", dataset_name, max_samples)
    dataset = load_dpo_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    dataloader = create_dpo_dataloader(dataset, batch_size=batch_size)

    # ---------------------------------------------------------------
    # 7. Optimizer (LoRA params only)
    # ---------------------------------------------------------------
    lr: float = float(training_cfg.get("learning_rate", 5e-5))
    weight_decay: float = float(training_cfg.get("weight_decay", 0.01))

    trainable_params = [p for p in target_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # ---------------------------------------------------------------
    # 8. Scheduler
    # ---------------------------------------------------------------
    num_epochs: int = training_cfg.get("num_epochs", 1)
    grad_accum_steps: int = training_cfg.get("gradient_accumulation_steps", 8)
    warmup_ratio: float = float(training_cfg.get("warmup_ratio", 0.1))

    total_batches = len(dataloader) * num_epochs
    total_optim_steps = total_batches // grad_accum_steps
    num_warmup_steps = int(warmup_ratio * total_optim_steps)

    scheduler = _create_scheduler(optimizer, num_warmup_steps, total_optim_steps)

    log.info(
        "Training plan — epochs: %d, batches/epoch: %d, grad_accum: %d, "
        "optimizer_steps: %d, warmup_steps: %d, lr: %.2e",
        num_epochs, len(dataloader), grad_accum_steps,
        total_optim_steps, num_warmup_steps, lr,
    )

    # ---------------------------------------------------------------
    # 9. Training loop
    # ---------------------------------------------------------------
    log_every: int = logging_cfg.get("log_every", 10)
    save_ckpt_every: int | None = logging_cfg.get("save_checkpoints_every", None)
    max_grad_norm: float = float(training_cfg.get("max_grad_norm", 1.0))

    all_metrics: list[dict] = []
    global_step: int = 0
    optim_step: int = 0
    running_dpo_loss: float = 0.0
    running_spec_loss: float = 0.0
    running_accept_proxy: float = 0.0
    running_pref_reward: float = 0.0
    running_rej_reward: float = 0.0
    running_margin: float = 0.0
    running_count: int = 0

    target_model.train()
    step_times: list[float] = []

    for epoch in range(num_epochs):
        log.info("=== Epoch %d / %d ===", epoch + 1, num_epochs)

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            chosen_ids = batch["chosen_input_ids"].to(target_device)
            chosen_mask = batch["chosen_attention_mask"].to(target_device)
            chosen_labels = batch["chosen_labels"].to(target_device)
            rejected_ids = batch["rejected_input_ids"].to(target_device)
            rejected_mask = batch["rejected_attention_mask"].to(target_device)
            rejected_labels = batch["rejected_labels"].to(target_device)

            # --- Reference model log-probs (no grad) ---
            with torch.no_grad():
                if reference_free:
                    # Use target model with adapters disabled
                    target_model.disable_adapter_layers()
                    target_model.eval()
                    if use_bf16 and target_device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            ref_chosen_out = target_model(
                                input_ids=chosen_ids, attention_mask=chosen_mask,
                            )
                            ref_rejected_out = target_model(
                                input_ids=rejected_ids, attention_mask=rejected_mask,
                            )
                    else:
                        ref_chosen_out = target_model(
                            input_ids=chosen_ids, attention_mask=chosen_mask,
                        )
                        ref_rejected_out = target_model(
                            input_ids=rejected_ids, attention_mask=rejected_mask,
                        )
                    target_model.enable_adapter_layers()
                    target_model.train()
                else:
                    if use_bf16 and target_device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            ref_chosen_out = ref_model(
                                input_ids=chosen_ids, attention_mask=chosen_mask,
                            )
                            ref_rejected_out = ref_model(
                                input_ids=rejected_ids, attention_mask=rejected_mask,
                            )
                    else:
                        ref_chosen_out = ref_model(
                            input_ids=chosen_ids, attention_mask=chosen_mask,
                        )
                        ref_rejected_out = ref_model(
                            input_ids=rejected_ids, attention_mask=rejected_mask,
                        )

                ref_chosen_logps = _compute_per_token_logps(
                    ref_chosen_out.logits, chosen_labels,
                )
                ref_rejected_logps = _compute_per_token_logps(
                    ref_rejected_out.logits, rejected_labels,
                )

            # --- Target model forward pass (with grad) ---
            if use_bf16 and target_device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    target_chosen_out = target_model(
                        input_ids=chosen_ids, attention_mask=chosen_mask,
                    )
                    target_rejected_out = target_model(
                        input_ids=rejected_ids, attention_mask=rejected_mask,
                    )
            else:
                target_chosen_out = target_model(
                    input_ids=chosen_ids, attention_mask=chosen_mask,
                )
                target_rejected_out = target_model(
                    input_ids=rejected_ids, attention_mask=rejected_mask,
                )

            # --- DPO loss ---
            dpo_result = compute_dpo_loss(
                target_chosen_logits=target_chosen_out.logits,
                target_rejected_logits=target_rejected_out.logits,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                beta=beta,
            )

            total_loss = dpo_result["dpo_loss"]

            # --- Spec loss (on chosen sequence, only if lambda > 0) ---
            spec_loss_val: float = 0.0
            accept_proxy_val: float = 0.0

            if lam > 0 and draft_model is not None:
                with torch.no_grad():
                    if use_bf16 and draft_device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            draft_out = draft_model(
                                input_ids=chosen_ids.to(draft_device),
                                attention_mask=chosen_mask.to(draft_device),
                            )
                    else:
                        draft_out = draft_model(
                            input_ids=chosen_ids.to(draft_device),
                            attention_mask=chosen_mask.to(draft_device),
                        )
                    draft_logits = draft_out.logits.to(target_device)

                spec_kl = _compute_spec_kl(
                    target_chosen_out.logits, draft_logits, chosen_mask, temperature,
                )
                total_loss = total_loss + lam * spec_kl
                spec_loss_val = spec_kl.detach().item()

                # Acceptance proxy for monitoring
                accept_proxy_val = _compute_acceptance_proxy(
                    target_chosen_out.logits, draft_logits, chosen_mask, temperature,
                )

            # --- Backward ---
            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()

            # --- Accumulate running stats ---
            running_dpo_loss += dpo_result["dpo_loss"].detach().item()
            running_spec_loss += spec_loss_val
            running_accept_proxy += accept_proxy_val
            running_pref_reward += dpo_result["preferred_reward"]
            running_rej_reward += dpo_result["rejected_reward"]
            running_margin += dpo_result["reward_margin"]
            running_count += 1

            global_step += 1

            # --- Optimizer step ---
            if global_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                step_end = time.time()
                step_times.append(step_end - step_start)

                # --- Logging ---
                if optim_step % log_every == 0 and running_count > 0:
                    n = running_count
                    current_lr = scheduler.get_last_lr()[0]

                    metric_entry = {
                        "epoch": epoch + 1,
                        "step": optim_step,
                        "dpo_loss": round(running_dpo_loss / n, 6),
                        "spec_loss": round(running_spec_loss / n, 6),
                        "acceptance_proxy": round(running_accept_proxy / n, 6),
                        "preferred_reward": round(running_pref_reward / n, 6),
                        "rejected_reward": round(running_rej_reward / n, 6),
                        "reward_margin": round(running_margin / n, 6),
                        "lr": current_lr,
                    }
                    all_metrics.append(metric_entry)

                    log.info(
                        "Step %d/%d | dpo=%.4f | spec=%.4f | accept=%.4f | "
                        "margin=%.4f | lr=%.2e",
                        optim_step, total_optim_steps,
                        running_dpo_loss / n, running_spec_loss / n,
                        running_accept_proxy / n, running_margin / n,
                        current_lr,
                    )

                    if len(step_times) >= 5 and optim_step <= 5 * log_every:
                        avg_step_time = sum(step_times) / len(step_times)
                        remaining = total_optim_steps - optim_step
                        eta_min = avg_step_time * remaining / 60.0
                        log.info(
                            "ETA: %.1f min (%.2f s/step)", eta_min, avg_step_time,
                        )

                    running_dpo_loss = 0.0
                    running_spec_loss = 0.0
                    running_accept_proxy = 0.0
                    running_pref_reward = 0.0
                    running_rej_reward = 0.0
                    running_margin = 0.0
                    running_count = 0

                # --- Checkpoint ---
                if save_ckpt_every is not None and optim_step % save_ckpt_every == 0:
                    save_checkpoint(target_model, output_dir, step=optim_step)

    # ---------------------------------------------------------------
    # 10. Post-training
    # ---------------------------------------------------------------
    log.info("Training complete. Saving final checkpoint and metrics.")
    save_checkpoint(target_model, output_dir, step=None)
    save_metrics(all_metrics, output_dir)
    tokenizer.save_pretrained(f"{output_dir}/final/tokenizer")
    log.info("Tokenizer saved to %s/final/tokenizer", output_dir)

    total_time = sum(step_times)
    log.info(
        "Finished %d optimizer steps in %.1f minutes.", optim_step, total_time / 60.0,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils import get_devices, load_config, parse_args

    args = parse_args()
    config = load_config(args.config, args.overrides)

    if args.domain is not None:
        config["domain"] = args.domain
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    target_device, draft_device = get_devices(args.target_device, args.draft_device)
    train_dpo(config, target_device, draft_device)
