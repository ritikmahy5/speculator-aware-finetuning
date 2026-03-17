"""Measure argmax agreement between target and draft models.

For each position in the evaluation prompts, checks whether
``argmax(target_logits) == argmax(draft_logits)``.  This directly measures
the mechanism behind speculative decoding acceptance — when the models agree
on the top-1 token, the draft is always accepted.

Usage::

    python -m src.measure_argmax \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --draft_model  meta-llama/Llama-3.2-1B-Instruct \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output results/argmax_agreement.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("specaware.measure_argmax")


def measure_argmax_agreement(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_prompts: list[str],
    target_device: torch.device,
    draft_device: torch.device,
    max_seq_len: int = 512,
) -> dict:
    """Measure per-position argmax agreement between target and draft.

    Args:
        target_model: Target language model (possibly with LoRA adapter).
        draft_model: Frozen draft language model.
        tokenizer: Shared tokenizer.
        eval_prompts: List of prompt strings to evaluate.
        target_device: Device for target model.
        draft_device: Device for draft model.
        max_seq_len: Maximum sequence length for tokenization.

    Returns:
        Dictionary with agreement statistics.
    """
    target_model.eval()
    draft_model.eval()

    per_prompt_agreement = []
    per_prompt_top5_overlap = []
    total_agree = 0
    total_positions = 0

    with torch.no_grad():
        for idx, prompt in enumerate(eval_prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )
            seq_len = inputs["input_ids"].size(1)
            if seq_len < 2:
                continue

            target_inputs = {k: v.to(target_device) for k, v in inputs.items()}
            draft_inputs = {k: v.to(draft_device) for k, v in inputs.items()}

            target_out = target_model(**target_inputs)
            draft_out = draft_model(**draft_inputs)

            # Compare logits at positions [0..seq_len-2] which predict tokens [1..seq_len-1]
            target_logits = target_out.logits[0, :-1, :]  # (seq_len-1, vocab)
            draft_logits = draft_out.logits[0, :-1, :].to(target_device)

            # Handle vocab size mismatch
            min_vocab = min(target_logits.size(-1), draft_logits.size(-1))
            target_logits = target_logits[:, :min_vocab]
            draft_logits = draft_logits[:, :min_vocab]

            n_pos = target_logits.size(0)

            # Top-1 agreement
            target_top1 = target_logits.argmax(dim=-1)  # (n_pos,)
            draft_top1 = draft_logits.argmax(dim=-1)
            agree = (target_top1 == draft_top1).sum().item()

            # Top-5 overlap: |intersection of top5_target and top5_draft| / 5
            target_top5 = target_logits.topk(5, dim=-1).indices  # (n_pos, 5)
            draft_top5 = draft_logits.topk(5, dim=-1).indices
            # For each position, count overlap
            overlap_per_pos = []
            for p in range(n_pos):
                t_set = set(target_top5[p].tolist())
                d_set = set(draft_top5[p].tolist())
                overlap_per_pos.append(len(t_set & d_set) / 5.0)
            mean_top5_overlap = sum(overlap_per_pos) / len(overlap_per_pos)

            prompt_agreement = agree / n_pos
            per_prompt_agreement.append(prompt_agreement)
            per_prompt_top5_overlap.append(mean_top5_overlap)
            total_agree += agree
            total_positions += n_pos

            if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_prompts):
                logger.info(
                    "Prompt %d/%d | agreement=%.4f | top5_overlap=%.4f | running=%.4f",
                    idx + 1,
                    len(eval_prompts),
                    prompt_agreement,
                    mean_top5_overlap,
                    total_agree / max(total_positions, 1),
                )

    overall_agreement = total_agree / max(total_positions, 1)
    agreement_std = float(torch.tensor(per_prompt_agreement).std().item()) if len(per_prompt_agreement) > 1 else 0.0
    top5_mean = sum(per_prompt_top5_overlap) / max(len(per_prompt_top5_overlap), 1)
    top5_std = float(torch.tensor(per_prompt_top5_overlap).std().item()) if len(per_prompt_top5_overlap) > 1 else 0.0

    results = {
        "argmax_agreement": overall_agreement,
        "argmax_agreement_std": agreement_std,
        "top5_overlap": top5_mean,
        "top5_overlap_std": top5_std,
        "per_prompt_agreement": per_prompt_agreement,
        "per_prompt_top5_overlap": per_prompt_top5_overlap,
        "num_prompts": len(per_prompt_agreement),
        "total_positions": total_positions,
    }

    logger.info(
        "Final: argmax_agreement=%.4f +/- %.4f | top5_overlap=%.4f +/- %.4f | total_positions=%d",
        overall_agreement, agreement_std, top5_mean, top5_std, total_positions,
    )
    return results


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from measure_acceptance.py)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer, optionally with a LoRA adapter."""
    logger.info("Loading model %s onto %s", model_name, device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs: dict = {"torch_dtype": dtype, "device_map": {"": device}}
    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if adapter_path is not None:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("LoRA adapter loaded successfully")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_prompts(prompts_file: str, domain: Optional[str] = None) -> list[str]:
    """Load evaluation prompts from a YAML file."""
    with open(prompts_file, "r") as f:
        prompts_data = yaml.safe_load(f)
    if domain is not None:
        key = domain if domain in prompts_data else f"{domain}_prompts"
        if key not in prompts_data:
            raise ValueError(f"Domain '{domain}' not found in {prompts_file}.")
        prompts = prompts_data[key]
    else:
        prompts = []
        for key in prompts_data:
            if isinstance(prompts_data[key], list):
                prompts.extend(prompts_data[key])
    logger.info("Loaded %d prompts from %s (domain=%s)", len(prompts), prompts_file, domain or "all")
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure argmax agreement between target and draft models")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--draft_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default="configs/eval_prompts.yaml")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/argmax_agreement.json")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--target_device", type=str, default=None)
    parser.add_argument("--draft_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    torch.manual_seed(args.seed)

    target_device = torch.device(args.target_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    draft_device = torch.device(args.draft_device or target_device)

    target_model, tokenizer = _load_model_and_tokenizer(args.target_model, target_device, args.adapter_path)
    draft_model, _ = _load_model_and_tokenizer(args.draft_model, draft_device)

    prompts = _load_prompts(args.prompts_file, args.domain)
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]

    results = measure_argmax_agreement(
        target_model, draft_model, tokenizer, prompts,
        target_device, draft_device, args.max_seq_len,
    )

    results["target_model"] = args.target_model
    results["draft_model"] = args.draft_model
    results["adapter_path"] = args.adapter_path
    results["domain"] = args.domain
    results["seed"] = args.seed

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
