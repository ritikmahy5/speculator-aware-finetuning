"""Joint training script for speculator-aware fine-tuning with two LoRA adapters.

Loads a target LLM and a draft model, applies LoRA to BOTH, then jointly trains:
  - Target: task loss (cross-entropy on domain data)
  - Draft: distillation loss (KL/overlap/soft_accept toward target's distribution)

This is the core training loop for EXP-8 (joint fine-tuning). Unlike train.py
where the draft is frozen, here both models are trainable with independent
optimizers, schedulers, and backward passes.

Usage:
    python -m src.train_joint --config configs/exp8_joint.yaml
    python -m src.train_joint --config configs/exp8_joint.yaml --draft_loss.type overlap
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import create_dataloader, load_dataset_for_domain
from src.utils import (
    get_devices,
    load_config,
    parse_args,
    save_config,
    save_dual_checkpoint,
    save_metrics,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("specaware")


# ---------------------------------------------------------------------------
# Helper: linear-warmup + cosine-decay scheduler
# ---------------------------------------------------------------------------

def _create_scheduler(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a linear-warmup then cosine-decay learning-rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps with linear ramp.
        num_training_steps: Total number of training steps (warmup + decay).

    Returns:
        A LambdaLR scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Helper: count parameters
# ---------------------------------------------------------------------------

def _log_param_counts(model: Any, label: str = "Model") -> None:
    """Log total and trainable parameter counts for the model.

    Args:
        model: A PyTorch / PEFT model.
        label: Human-readable name for log messages.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "%s parameters — total: %s, trainable: %s (%.2f%%)",
        label,
        f"{total_params:,}",
        f"{trainable_params:,}",
        100.0 * trainable_params / total_params if total_params > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main joint training function
# ---------------------------------------------------------------------------

def train(config: dict, target_device: torch.device, draft_device: torch.device) -> None:
    """Run joint LoRA fine-tuning with two adapters and two optimizers.

    Pipeline:
      1. Setup (seed, logging, config persistence)
      2. Tokenizer and model loading (target with LoRA, draft with LoRA)
      3. Dataset and dataloader creation
      4. Two optimizers and two schedulers
      5. Training loop: target task loss + draft distillation loss (independent backward)
      6. Dual checkpoint and metric saving

    Args:
        config: Fully-merged experiment config dictionary.
        target_device: Device for the target model (e.g. ``torch.device("cuda:0")``).
        draft_device: Device for the draft model (e.g. ``torch.device("cuda:1")``).
    """
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    training_cfg: dict = config.get("training", {})
    target_lora_cfg: dict = config.get("target_lora", config.get("lora", {}))
    draft_lora_cfg: dict = config.get("draft_lora", {})
    draft_loss_cfg: dict = config.get("draft_loss", {})
    logging_cfg: dict = config.get("logging", {})

    seed: int = training_cfg.get("seed", 42)
    set_seed(seed)

    output_dir: str = config.get("output_dir", "results/default_run")
    log = setup_logging(output_dir)
    save_config(config, output_dir)

    log.info("Experiment: %s", config.get("experiment_id", "N/A"))
    log.info("Mode: JOINT TRAINING (target task loss + draft distillation)")
    log.info("Target device: %s | Draft device: %s", target_device, draft_device)

    use_bf16: bool = training_cfg.get("bf16", True)

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
    # 3. Load target model + LoRA (trainable)
    # ---------------------------------------------------------------
    log.info("Loading target model: %s", target_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=torch.bfloat16,
    )
    target_model.to(target_device)

    target_lora = LoraConfig(
        r=target_lora_cfg.get("rank", 16),
        lora_alpha=target_lora_cfg.get("alpha", 32),
        lora_dropout=target_lora_cfg.get("dropout", 0.05),
        target_modules=target_lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    target_model = get_peft_model(target_model, target_lora)
    target_model.gradient_checkpointing_enable()
    _log_param_counts(target_model, "Target")

    # ---------------------------------------------------------------
    # 4. Load draft model + LoRA (trainable, NOT frozen)
    # ---------------------------------------------------------------
    log.info("Loading draft model: %s (with LoRA — trainable)", draft_name)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_name,
        torch_dtype=torch.bfloat16,
    )
    draft_model.to(draft_device)

    draft_lora = LoraConfig(
        r=draft_lora_cfg.get("rank", 8),
        lora_alpha=draft_lora_cfg.get("alpha", 16),
        lora_dropout=draft_lora_cfg.get("dropout", 0.05),
        target_modules=draft_lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    draft_model = get_peft_model(draft_model, draft_lora)
    _log_param_counts(draft_model, "Draft")

    # ---------------------------------------------------------------
    # 5. Load data
    # ---------------------------------------------------------------
    domain: str = config.get("domain", "code")
    max_samples: int = training_cfg.get("max_samples", 10000)
    max_seq_len: int = training_cfg.get("max_seq_len", 1024)
    batch_size: int = training_cfg.get("batch_size", 4)

    log.info("Loading dataset: domain=%s, max_samples=%d", domain, max_samples)
    dataset = load_dataset_for_domain(
        domain=domain,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    dataloader = create_dataloader(dataset, batch_size=batch_size)

    # ---------------------------------------------------------------
    # 6. Two optimizers (target and draft LoRA params)
    # ---------------------------------------------------------------
    target_lr: float = float(target_lora_cfg.get("learning_rate", 2e-4))
    draft_lr: float = float(draft_lora_cfg.get("learning_rate", 5e-4))
    target_wd: float = float(target_lora_cfg.get(
        "weight_decay", training_cfg.get("weight_decay", 0.01)
    ))
    draft_wd: float = float(draft_lora_cfg.get(
        "weight_decay", training_cfg.get("weight_decay", 0.01)
    ))

    target_trainable = [p for p in target_model.parameters() if p.requires_grad]
    draft_trainable = [p for p in draft_model.parameters() if p.requires_grad]

    optimizer_target = AdamW(target_trainable, lr=target_lr, weight_decay=target_wd)
    optimizer_draft = AdamW(draft_trainable, lr=draft_lr, weight_decay=draft_wd)

    # ---------------------------------------------------------------
    # 7. Two schedulers
    # ---------------------------------------------------------------
    num_epochs: int = training_cfg.get("num_epochs", 1)
    grad_accum_steps: int = training_cfg.get("gradient_accumulation_steps", 4)
    warmup_ratio: float = float(training_cfg.get("warmup_ratio", 0.05))

    total_batches = len(dataloader) * num_epochs
    total_optim_steps = total_batches // grad_accum_steps
    num_warmup_steps = int(warmup_ratio * total_optim_steps)

    scheduler_target = _create_scheduler(optimizer_target, num_warmup_steps, total_optim_steps)
    scheduler_draft = _create_scheduler(optimizer_draft, num_warmup_steps, total_optim_steps)

    log.info(
        "Training plan — epochs: %d, optimizer_steps: %d, target_lr: %.2e, draft_lr: %.2e",
        num_epochs, total_optim_steps, target_lr, draft_lr,
    )

    # ---------------------------------------------------------------
    # 8. Training loop
    # ---------------------------------------------------------------
    draft_loss_type: str = draft_loss_cfg.get("type", "kl")
    draft_loss_temp: float = float(draft_loss_cfg.get("temperature", 1.0))
    draft_loss_top_k: int | None = draft_loss_cfg.get("top_k", None)

    log_every: int = logging_cfg.get("log_every", 10)
    save_ckpt_every: int | None = logging_cfg.get("save_checkpoints_every", None)
    max_grad_norm: float = float(training_cfg.get("max_grad_norm", 1.0))

    log.info("Draft loss type: %s | temperature: %.2f", draft_loss_type, draft_loss_temp)

    all_metrics: list[dict] = []
    global_step: int = 0
    optim_step: int = 0
    running: dict[str, float] = {
        "task_loss": 0.0,
        "draft_loss": 0.0,
        "accept_proxy": 0.0,
        "count": 0,
    }

    target_model.train()
    draft_model.train()
    step_times: list[float] = []

    for epoch in range(num_epochs):
        log.info("=== Epoch %d / %d ===", epoch + 1, num_epochs)

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # --- Target forward + task loss ---
            if use_bf16 and target_device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    target_out = target_model(
                        input_ids=input_ids.to(target_device),
                        attention_mask=attention_mask.to(target_device),
                    )
                    target_logits = target_out.logits
            else:
                target_out = target_model(
                    input_ids=input_ids.to(target_device),
                    attention_mask=attention_mask.to(target_device),
                )
                target_logits = target_out.logits

            # Task loss: standard cross-entropy on target
            shift_logits = target_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].to(target_device).contiguous()
            task_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # --- Draft forward ---
            if use_bf16 and draft_device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    draft_out = draft_model(
                        input_ids=input_ids.to(draft_device),
                        attention_mask=attention_mask.to(draft_device),
                    )
                    draft_logits = draft_out.logits
            else:
                draft_out = draft_model(
                    input_ids=input_ids.to(draft_device),
                    attention_mask=attention_mask.to(draft_device),
                )
                draft_logits = draft_out.logits

            # --- Draft loss (distillation): draft chases target ---
            # CRITICAL: target_logits.detach() so target gradients do NOT flow
            # through the draft loss. The two backward passes are independent.
            if draft_loss_type == "kl":
                # KL(draft || target.detach())
                target_probs = torch.nn.functional.softmax(
                    target_logits.detach().to(draft_device)[..., :-1, :] / draft_loss_temp,
                    dim=-1,
                )
                draft_log_probs = torch.nn.functional.log_softmax(
                    draft_logits[..., :-1, :] / draft_loss_temp,
                    dim=-1,
                )
                # Handle vocab size mismatch between target and draft
                min_vocab = min(target_probs.size(-1), draft_log_probs.size(-1))
                target_probs = target_probs[..., :min_vocab]
                draft_log_probs = draft_log_probs[..., :min_vocab]

                shift_mask = attention_mask[..., :-1].to(draft_device).float()
                kl_per_pos = torch.nn.functional.kl_div(
                    draft_log_probs, target_probs, reduction="none"
                ).sum(dim=-1)
                draft_loss = (kl_per_pos * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)

            elif draft_loss_type == "overlap":
                from src.acceptance_proxy_loss import compute_overlap_loss

                proxy_result = compute_overlap_loss(
                    target_logits=target_logits.detach().to(draft_device),
                    draft_logits=draft_logits,
                    attention_mask=attention_mask.to(draft_device),
                    top_k=draft_loss_top_k or 50,
                    temperature=draft_loss_temp,
                )
                draft_loss = proxy_result["overlap_loss"]

            elif draft_loss_type == "soft_accept":
                from src.acceptance_proxy_loss import compute_soft_accept_loss

                proxy_result = compute_soft_accept_loss(
                    target_logits=target_logits.detach().to(draft_device),
                    draft_logits=draft_logits,
                    attention_mask=attention_mask.to(draft_device),
                    tau=draft_loss_temp,
                    num_samples=4,
                    temperature=1.0,
                )
                draft_loss = proxy_result["accept_loss"]

            else:
                raise ValueError(f"Unknown draft_loss.type: {draft_loss_type}")

            # --- Backward target (frees target computation graph) ---
            scaled_target = task_loss / grad_accum_steps
            scaled_target.backward()

            # --- Backward draft (independent graph, no retain_graph needed) ---
            scaled_draft = draft_loss / grad_accum_steps
            scaled_draft.backward()

            # --- Acceptance proxy for monitoring (no gradients needed) ---
            with torch.no_grad():
                target_probs_mon = torch.nn.functional.softmax(
                    target_logits[..., :-1, :].detach(), dim=-1
                )
                draft_probs_mon = torch.nn.functional.softmax(
                    draft_logits[..., :-1, :].detach().to(target_device), dim=-1
                )
                min_v = min(target_probs_mon.size(-1), draft_probs_mon.size(-1))
                draft_argmax = draft_probs_mon[..., :min_v].argmax(dim=-1)
                p_target_at_draft = torch.gather(
                    target_probs_mon[..., :min_v],
                    dim=-1,
                    index=draft_argmax.unsqueeze(-1),
                ).squeeze(-1)
                p_draft_at_draft = torch.gather(
                    draft_probs_mon[..., :min_v],
                    dim=-1,
                    index=draft_argmax.unsqueeze(-1),
                ).squeeze(-1)
                accept = torch.min(
                    torch.ones_like(p_target_at_draft),
                    p_target_at_draft / (p_draft_at_draft + 1e-10),
                )
                mask_mon = attention_mask[..., :-1].to(target_device).float()
                accept_proxy = (accept * mask_mon).sum() / mask_mon.sum().clamp(min=1.0)

            # --- Accumulate stats ---
            running["task_loss"] += task_loss.detach().item()
            running["draft_loss"] += draft_loss.detach().item()
            running["accept_proxy"] += accept_proxy.item()
            running["count"] += 1
            global_step += 1

            # --- Optimizer step (every grad_accum_steps micro-batches) ---
            if global_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(target_trainable, max_grad_norm)
                torch.nn.utils.clip_grad_norm_(draft_trainable, max_grad_norm)
                optimizer_target.step()
                optimizer_draft.step()
                scheduler_target.step()
                scheduler_draft.step()
                optimizer_target.zero_grad()
                optimizer_draft.zero_grad()
                optim_step += 1

                step_end = time.time()
                step_times.append(step_end - step_start)

                # --- Logging ---
                if optim_step % log_every == 0 and running["count"] > 0:
                    c = running["count"]
                    avg_task = running["task_loss"] / c
                    avg_draft = running["draft_loss"] / c
                    avg_accept = running["accept_proxy"] / c

                    # LoRA weight norms for stability monitoring
                    target_norm = sum(p.norm().item() ** 2 for p in target_trainable) ** 0.5
                    draft_norm = sum(p.norm().item() ** 2 for p in draft_trainable) ** 0.5

                    metric_entry = {
                        "epoch": epoch + 1,
                        "step": optim_step,
                        "task_loss": round(avg_task, 6),
                        "draft_loss": round(avg_draft, 6),
                        "acceptance_proxy": round(avg_accept, 6),
                        "lr_target": scheduler_target.get_last_lr()[0],
                        "lr_draft": scheduler_draft.get_last_lr()[0],
                        "target_lora_norm": round(target_norm, 4),
                        "draft_lora_norm": round(draft_norm, 4),
                    }
                    all_metrics.append(metric_entry)

                    log.info(
                        "Step %d/%d | task=%.4f | draft=%.4f | accept=%.4f | "
                        "t_norm=%.2f | d_norm=%.2f",
                        optim_step, total_optim_steps, avg_task, avg_draft,
                        avg_accept, target_norm, draft_norm,
                    )

                    # Estimate remaining time after a few steps
                    if len(step_times) >= 5 and optim_step <= 5 * log_every:
                        avg_step_time = sum(step_times) / len(step_times)
                        remaining_steps = total_optim_steps - optim_step
                        eta_minutes = (avg_step_time * remaining_steps) / 60.0
                        log.info(
                            "Estimated remaining time: %.1f minutes (%.2f s/step)",
                            eta_minutes, avg_step_time,
                        )

                    running = {
                        "task_loss": 0.0,
                        "draft_loss": 0.0,
                        "accept_proxy": 0.0,
                        "count": 0,
                    }

                # --- Save intermediate checkpoint ---
                if save_ckpt_every is not None and optim_step % save_ckpt_every == 0:
                    save_dual_checkpoint(target_model, draft_model, output_dir, step=optim_step)

    # ---------------------------------------------------------------
    # 9. Post-training: save final checkpoint and metrics
    # ---------------------------------------------------------------
    log.info("Training complete. Saving final checkpoint and metrics.")
    save_dual_checkpoint(target_model, draft_model, output_dir, step=None)
    save_metrics(all_metrics, output_dir)

    # Save tokenizer alongside the final adapters
    tokenizer.save_pretrained(f"{output_dir}/final/tokenizer")
    log.info("Tokenizer saved to %s/final/tokenizer", output_dir)

    total_time = sum(step_times)
    log.info(
        "Finished %d optimizer steps in %.1f minutes.",
        optim_step, total_time / 60.0,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.overrides)

    # Apply direct CLI overrides for domain and output_dir
    if args.domain is not None:
        config["domain"] = args.domain
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    target_device, draft_device = get_devices(args.target_device, args.draft_device)

    train(config, target_device, draft_device)
