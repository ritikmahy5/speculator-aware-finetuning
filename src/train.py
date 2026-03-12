"""Main training script for speculator-aware fine-tuning with LoRA.

Loads a target LLM and (optionally) a frozen draft model, then fine-tunes
the target with LoRA while adding a KL-divergence regularization term that
keeps the target distribution close to the draft model.  This preserves
speculative-decoding acceptance rates after domain-specific fine-tuning.

Usage:
    python src/train.py --config configs/exp3_spec_aware.yaml
    python src/train.py --config configs/exp3_spec_aware.yaml --spec_loss.lam 0.05
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import create_dataloader, load_dataset_for_domain
from src.spec_loss import compute_spec_loss
from src.utils import (
    get_devices,
    load_config,
    parse_args,
    save_checkpoint,
    save_config,
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
    import math

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

def _log_param_counts(model: Any) -> None:
    """Log total and trainable parameter counts for the model.

    Args:
        model: A PyTorch / PEFT model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters — total: %s, trainable: %s (%.2f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        100.0 * trainable_params / total_params if total_params > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict, target_device: torch.device, draft_device: torch.device) -> None:
    """Run speculator-aware LoRA fine-tuning.

    This function orchestrates the full training pipeline:
      1. Setup (seed, logging, config persistence)
      2. Tokenizer and model loading (target with LoRA, draft frozen)
      3. Dataset and dataloader creation
      4. Optimizer and scheduler initialization
      5. Training loop with optional speculator-aware loss
      6. Checkpoint and metric saving

    Args:
        config: Fully-merged experiment config dictionary.
        target_device: Device for the target model (e.g. ``torch.device("cuda:0")``).
        draft_device: Device for the draft model (e.g. ``torch.device("cuda:1")``).
    """
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    training_cfg: dict = config.get("training", {})
    lora_cfg: dict = config.get("lora", {})
    spec_cfg: dict = config.get("spec_loss", {})
    logging_cfg: dict = config.get("logging", {})

    seed: int = training_cfg.get("seed", 42)
    set_seed(seed)

    output_dir: str = config.get("output_dir", "results/default_run")
    log = setup_logging(output_dir)
    save_config(config, output_dir)

    log.info("Experiment: %s", config.get("experiment_id", "N/A"))
    log.info("Target device: %s | Draft device: %s", target_device, draft_device)

    lam: float = float(spec_cfg.get("lam", 0.0))
    loss_type: str = spec_cfg.get("loss_type", "kl")
    temperature: float = float(spec_cfg.get("temperature", 1.0))
    top_k: int | None = spec_cfg.get("top_k", None)

    use_bf16: bool = training_cfg.get("bf16", True)
    log.info("Lambda: %.4f | Loss type: %s | bf16: %s", lam, loss_type, use_bf16)

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

    # Apply LoRA
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
    # 4. Load draft model (only if lam > 0)
    # ---------------------------------------------------------------
    draft_model: AutoModelForCausalLM | None = None

    if lam > 0:
        log.info("Loading draft model: %s (frozen)", draft_name)
        same_device = target_device == draft_device
        draft_quantize_4bit: bool = config.get("draft_quantize_4bit", False) and same_device

        if draft_quantize_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft_name,
                quantization_config=bnb_config,
                device_map={"": draft_device},
                torch_dtype=torch.bfloat16,
            )
        else:
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
        log.info("Lambda=0 — skipping draft model loading (standard fine-tuning).")

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
    # 6. Optimizer (LoRA params only)
    # ---------------------------------------------------------------
    lr: float = float(training_cfg.get("learning_rate", 2e-4))
    weight_decay: float = float(training_cfg.get("weight_decay", 0.01))

    trainable_params = [p for p in target_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # ---------------------------------------------------------------
    # 7. Scheduler
    # ---------------------------------------------------------------
    num_epochs: int = training_cfg.get("num_epochs", 1)
    grad_accum_steps: int = training_cfg.get("gradient_accumulation_steps", 4)
    warmup_ratio: float = float(training_cfg.get("warmup_ratio", 0.05))

    total_batches = len(dataloader) * num_epochs
    total_optim_steps = total_batches // grad_accum_steps
    num_warmup_steps = int(warmup_ratio * total_optim_steps)

    scheduler = _create_scheduler(optimizer, num_warmup_steps, total_optim_steps)

    log.info(
        "Training plan — epochs: %d, batches/epoch: %d, grad_accum: %d, "
        "optimizer_steps: %d, warmup_steps: %d, lr: %.2e",
        num_epochs,
        len(dataloader),
        grad_accum_steps,
        total_optim_steps,
        num_warmup_steps,
        lr,
    )

    # ---------------------------------------------------------------
    # 8. Training loop
    # ---------------------------------------------------------------
    log_every: int = logging_cfg.get("log_every", 10)
    save_ckpt_every: int | None = logging_cfg.get("save_checkpoints_every", None)
    max_grad_norm: float = float(training_cfg.get("max_grad_norm", 1.0))

    all_metrics: list[dict] = []
    global_step: int = 0  # counts micro-batches
    optim_step: int = 0   # counts optimizer updates
    running_task_loss: float = 0.0
    running_spec_loss: float = 0.0
    running_acceptance_proxy: float = 0.0
    running_count: int = 0

    target_model.train()
    step_times: list[float] = []

    for epoch in range(num_epochs):
        log.info("=== Epoch %d / %d ===", epoch + 1, num_epochs)

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            input_ids: torch.Tensor = batch["input_ids"].to(target_device)
            attention_mask: torch.Tensor = batch["attention_mask"].to(target_device)
            labels: torch.Tensor = batch["labels"].to(target_device)

            # --- Target forward pass (with autocast if bf16) ---
            if use_bf16 and target_device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    target_outputs = target_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    target_logits = target_outputs.logits
            else:
                target_outputs = target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                target_logits = target_outputs.logits

            # --- Draft forward pass (no grad, only if lam > 0) ---
            draft_logits: torch.Tensor | None = None
            if lam > 0 and draft_model is not None:
                with torch.no_grad():
                    if use_bf16 and draft_device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            draft_outputs = draft_model(
                                input_ids=input_ids.to(draft_device),
                                attention_mask=attention_mask.to(draft_device),
                            )
                    else:
                        draft_outputs = draft_model(
                            input_ids=input_ids.to(draft_device),
                            attention_mask=attention_mask.to(draft_device),
                        )
                    # Move draft logits to target device for loss computation
                    draft_logits = draft_outputs.logits.to(target_device)

            # --- Compute loss ---
            loss_dict = compute_spec_loss(
                target_logits=target_logits,
                draft_logits=draft_logits,
                labels=labels,
                attention_mask=attention_mask,
                lam=lam,
                loss_type=loss_type,
                temperature=temperature,
                top_k=top_k,
            )

            loss: torch.Tensor = loss_dict["loss"] / grad_accum_steps
            loss.backward()

            # --- Accumulate running stats (values are already floats) ---
            running_task_loss += loss_dict["task_loss"]
            running_spec_loss += loss_dict["spec_loss"]
            running_acceptance_proxy += loss_dict["acceptance_proxy"]
            running_count += 1

            global_step += 1

            # --- Optimizer step (every grad_accum_steps micro-batches) ---
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
                    avg_task = running_task_loss / running_count
                    avg_spec = running_spec_loss / running_count
                    avg_accept = running_acceptance_proxy / running_count
                    current_lr = scheduler.get_last_lr()[0]

                    metric_entry = {
                        "epoch": epoch + 1,
                        "step": optim_step,
                        "task_loss": round(avg_task, 6),
                        "spec_loss": round(avg_spec, 6),
                        "acceptance_proxy": round(avg_accept, 6),
                        "lr": current_lr,
                    }
                    all_metrics.append(metric_entry)

                    log.info(
                        "Step %d/%d | task_loss=%.4f | spec_loss=%.4f | "
                        "accept_proxy=%.4f | lr=%.2e",
                        optim_step,
                        total_optim_steps,
                        avg_task,
                        avg_spec,
                        avg_accept,
                        current_lr,
                    )

                    # Estimate remaining time after a few steps
                    if len(step_times) >= 5 and optim_step <= 5 * log_every:
                        avg_step_time = sum(step_times) / len(step_times)
                        remaining_steps = total_optim_steps - optim_step
                        eta_seconds = avg_step_time * remaining_steps
                        eta_minutes = eta_seconds / 60.0
                        log.info(
                            "Estimated remaining time: %.1f minutes (%.2f s/step)",
                            eta_minutes,
                            avg_step_time,
                        )

                    running_task_loss = 0.0
                    running_spec_loss = 0.0
                    running_acceptance_proxy = 0.0
                    running_count = 0

                # --- Save intermediate checkpoint ---
                if save_ckpt_every is not None and optim_step % save_ckpt_every == 0:
                    save_checkpoint(target_model, output_dir, step=optim_step)

    # ---------------------------------------------------------------
    # 9. Post-training: save final checkpoint and metrics
    # ---------------------------------------------------------------
    log.info("Training complete. Saving final checkpoint and metrics.")
    save_checkpoint(target_model, output_dir, step=None)
    save_metrics(all_metrics, output_dir)

    # Also save the tokenizer alongside the final adapter
    tokenizer.save_pretrained(f"{output_dir}/final/tokenizer")
    log.info("Tokenizer saved to %s/final/tokenizer", output_dir)

    total_time = sum(step_times)
    log.info(
        "Finished %d optimizer steps in %.1f minutes.",
        optim_step,
        total_time / 60.0,
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
