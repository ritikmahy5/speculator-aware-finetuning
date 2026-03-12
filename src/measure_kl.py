"""Measure KL, JS, and TV divergence between target and draft models.

Evaluates distributional divergence on a set of evaluation prompts.  Both
models run in eval mode under ``torch.no_grad()`` -- this is pure measurement,
not training.

Usage::

    python -m src.measure_kl \
        --target_model Qwen/Qwen2.5-7B-Instruct \
        --draft_model  Qwen/Qwen2.5-0.5B-Instruct \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output results/exp1/eval_kl.json
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

logger = logging.getLogger("specaware.measure_kl")

EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Divergence computation helpers
# ---------------------------------------------------------------------------

def _kl_per_position(
    target_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    draft_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Forward KL divergence KL(target || draft) per position.

    Args:
        target_probs: (seq_len, vocab_size) target probabilities.
        target_log_probs: (seq_len, vocab_size) target log probabilities.
        draft_log_probs: (seq_len, vocab_size) draft log probabilities.

    Returns:
        (seq_len,) KL divergence at each position.
    """
    draft_log_safe = torch.clamp(draft_log_probs, min=torch.log(torch.tensor(EPSILON)))
    return (target_probs * (target_log_probs - draft_log_safe)).sum(dim=-1)


def _reverse_kl_per_position(
    draft_probs: torch.Tensor,
    draft_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Reverse KL divergence KL(draft || target) per position.

    Args:
        draft_probs: (seq_len, vocab_size) draft probabilities.
        draft_log_probs: (seq_len, vocab_size) draft log probabilities.
        target_log_probs: (seq_len, vocab_size) target log probabilities.

    Returns:
        (seq_len,) reverse KL divergence at each position.
    """
    target_log_safe = torch.clamp(target_log_probs, min=torch.log(torch.tensor(EPSILON)))
    return (draft_probs * (draft_log_probs - target_log_safe)).sum(dim=-1)


def _js_per_position(
    target_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Jensen-Shannon divergence per position.

    JS = 0.5 * KL(target || m) + 0.5 * KL(draft || m), m = 0.5*(p+q).

    Args:
        target_probs: (seq_len, vocab_size).
        target_log_probs: (seq_len, vocab_size).
        draft_probs: (seq_len, vocab_size).
        draft_log_probs: (seq_len, vocab_size).

    Returns:
        (seq_len,) JS divergence at each position.
    """
    m = 0.5 * (target_probs + draft_probs)
    log_m = torch.log(m + EPSILON)
    kl_target_m = (target_probs * (target_log_probs - log_m)).sum(dim=-1)
    kl_draft_m = (draft_probs * (draft_log_probs - log_m)).sum(dim=-1)
    return 0.5 * kl_target_m + 0.5 * kl_draft_m


def _tv_per_position(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Total Variation distance per position.

    TV = 0.5 * sum(|p_target - p_draft|).

    Args:
        target_probs: (seq_len, vocab_size).
        draft_probs: (seq_len, vocab_size).

    Returns:
        (seq_len,) TV distance at each position.
    """
    return 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)


# Map divergence name -> callable
_DIVERGENCE_FNS = {
    "kl": "kl",
    "reverse_kl": "reverse_kl",
    "js": "js",
    "tv": "tv",
}


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

def measure_divergence(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_prompts: list[str],
    target_device: torch.device,
    draft_device: torch.device,
    divergence_types: list[str] | None = None,
    max_seq_len: int = 512,
) -> dict:
    """Measure divergence between target and draft model distributions.

    Both models are run in eval mode under ``torch.no_grad()``.  For each
    prompt the function tokenizes the text, runs both models, and computes
    the requested divergences at every token position, then averages.

    Args:
        target_model: The (possibly LoRA-adapted) target language model.
        draft_model: The frozen draft language model.
        tokenizer: Tokenizer shared by both models.
        eval_prompts: List of prompt strings to evaluate.
        target_device: Device the target model lives on.
        draft_device: Device the draft model lives on.
        divergence_types: List of divergence types to compute.  Supported:
            ``"kl"``, ``"reverse_kl"``, ``"js"``, ``"tv"``.
            Defaults to ``["kl", "js", "tv"]``.
        max_seq_len: Maximum sequence length (prompts are truncated).

    Returns:
        Dictionary with, for each divergence type ``t``:
            - ``{t}_mean``: mean divergence across all prompts and positions.
            - ``{t}_std``: standard deviation of per-prompt mean divergences.
            - ``{t}_per_prompt``: list of per-prompt mean divergences.
        Plus:
            - ``num_prompts``: number of prompts evaluated.
            - ``avg_seq_len``: average sequence length after tokenization.
    """
    if divergence_types is None:
        divergence_types = ["kl", "js", "tv"]

    for dt in divergence_types:
        if dt not in _DIVERGENCE_FNS:
            raise ValueError(
                f"Unknown divergence type '{dt}'. "
                f"Supported: {list(_DIVERGENCE_FNS.keys())}"
            )

    target_model.eval()
    draft_model.eval()

    # Accumulators: divergence_type -> list of per-prompt means
    per_prompt: dict[str, list[float]] = {dt: [] for dt in divergence_types}
    total_seq_len = 0

    for prompt_idx, prompt in enumerate(eval_prompts):
        # Tokenize
        encoding = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        input_ids = encoding["input_ids"]  # (1, seq_len)
        seq_len = input_ids.size(1)
        total_seq_len += seq_len

        if seq_len < 2:
            logger.warning(
                "Prompt %d too short (%d tokens), skipping.", prompt_idx, seq_len
            )
            continue

        # Forward passes (measurement only, no gradients anywhere)
        with torch.no_grad():
            target_out = target_model(input_ids=input_ids.to(target_device))
            draft_out = draft_model(input_ids=input_ids.to(draft_device))

        # Logits: (1, seq_len, vocab_size) -> squeeze batch dim
        target_logits = target_out.logits[0]  # (seq_len, vocab_size)
        draft_logits = draft_out.logits[0].to(target_device)  # move to same device

        # Convert to probs and log-probs
        target_probs = F.softmax(target_logits, dim=-1)
        target_log_probs = F.log_softmax(target_logits, dim=-1)
        draft_probs = F.softmax(draft_logits, dim=-1)
        draft_log_probs = F.log_softmax(draft_logits, dim=-1)

        # Compute each requested divergence
        for dt in divergence_types:
            if dt == "kl":
                div_vals = _kl_per_position(
                    target_probs, target_log_probs, draft_log_probs
                )
            elif dt == "reverse_kl":
                div_vals = _reverse_kl_per_position(
                    draft_probs, draft_log_probs, target_log_probs
                )
            elif dt == "js":
                div_vals = _js_per_position(
                    target_probs, target_log_probs, draft_probs, draft_log_probs
                )
            elif dt == "tv":
                div_vals = _tv_per_position(target_probs, draft_probs)
            else:
                raise ValueError(f"Unexpected divergence type: {dt}")

            # Mean over positions for this prompt
            prompt_mean = div_vals.mean().item()
            per_prompt[dt].append(prompt_mean)

        if (prompt_idx + 1) % 10 == 0 or (prompt_idx + 1) == len(eval_prompts):
            summary_parts = []
            for dt in divergence_types:
                running_mean = sum(per_prompt[dt]) / len(per_prompt[dt])
                summary_parts.append(f"{dt}={running_mean:.4f}")
            logger.info(
                "Prompt %d/%d  |  %s",
                prompt_idx + 1,
                len(eval_prompts),
                "  ".join(summary_parts),
            )

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    num_prompts = len(per_prompt[divergence_types[0]])
    avg_seq_len = total_seq_len / max(num_prompts, 1)

    results: dict = {
        "num_prompts": num_prompts,
        "avg_seq_len": avg_seq_len,
    }

    for dt in divergence_types:
        vals = per_prompt[dt]
        t = torch.tensor(vals) if vals else torch.tensor([0.0])
        results[f"{dt}_mean"] = t.mean().item()
        results[f"{dt}_std"] = t.std().item() if len(vals) > 1 else 0.0
        results[f"{dt}_per_prompt"] = vals

    logger.info("=" * 60)
    logger.info("DIVERGENCE MEASUREMENT SUMMARY")
    logger.info("=" * 60)
    for dt in divergence_types:
        logger.info(
            "  %-12s  mean=%.6f  std=%.6f",
            dt,
            results[f"{dt}_mean"],
            results[f"{dt}_std"],
        )
    logger.info("  num_prompts:  %d", num_prompts)
    logger.info("  avg_seq_len:  %.1f", avg_seq_len)
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    adapter_path: Optional[str] = None,
    load_in_8bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer, optionally with a LoRA adapter.

    Args:
        model_name: HuggingFace model ID or local path.
        device: Device to place the model on.
        adapter_path: Optional path to a PEFT LoRA adapter directory.
        load_in_8bit: Whether to load the model in 8-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info("Loading model %s onto %s", model_name, device)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs: dict = {"torch_dtype": dtype}

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["device_map"] = {"": device}

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
    """Load evaluation prompts from a YAML file.

    Args:
        prompts_file: Path to the YAML prompts file.
        domain: If provided, load only prompts for this domain.

    Returns:
        List of prompt strings.
    """
    with open(prompts_file, "r") as f:
        prompts_data = yaml.safe_load(f)

    if domain is not None:
        # Support both "code" and "code_prompts" key formats
        key = domain if domain in prompts_data else f"{domain}_prompts"
        if key not in prompts_data:
            raise ValueError(
                f"Domain '{domain}' not found in {prompts_file}. "
                f"Available: {list(prompts_data.keys())}"
            )
        prompts = prompts_data[key]
    else:
        prompts = []
        for key in prompts_data:
            if isinstance(prompts_data[key], list):
                prompts.extend(prompts_data[key])

    logger.info(
        "Loaded %d prompts from %s (domain=%s)",
        len(prompts),
        prompts_file,
        domain or "all",
    )
    return prompts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for divergence measurement.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Measure KL/JS/TV divergence between target and draft models",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for the target model.",
    )
    parser.add_argument(
        "--draft_model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for the draft model.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to a PEFT LoRA adapter directory for the target model.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="configs/eval_prompts.yaml",
        help="Path to YAML file containing evaluation prompts.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain subset to use from prompts file (code, medical, chat, mixed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_kl.json",
        help="Path to write the output JSON results.",
    )
    parser.add_argument(
        "--divergence_types",
        type=str,
        nargs="+",
        default=["kl", "js", "tv"],
        help="Divergence types to compute. Options: kl, reverse_kl, js, tv.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--target_device",
        type=str,
        default=None,
        help="Device for target model, e.g. 'cuda:0'.",
    )
    parser.add_argument(
        "--draft_device",
        type=str,
        default=None,
        help="Device for draft model, e.g. 'cuda:1'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load models, run divergence measurement, save results."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from src.utils import set_seed, get_devices

    set_seed(args.seed)
    target_device, draft_device = get_devices(args.target_device, args.draft_device)

    # Load models
    target_model, tokenizer = _load_model_and_tokenizer(
        args.target_model,
        target_device,
        adapter_path=args.adapter_path,
    )
    draft_model, _ = _load_model_and_tokenizer(
        args.draft_model,
        draft_device,
    )

    # Freeze draft model
    for param in draft_model.parameters():
        param.requires_grad = False

    # Load prompts
    prompts = _load_prompts(args.prompts_file, args.domain)

    # Run measurement
    results = measure_divergence(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        eval_prompts=prompts,
        target_device=target_device,
        draft_device=draft_device,
        divergence_types=args.divergence_types,
        max_seq_len=args.max_seq_len,
    )

    # Add metadata
    results["target_model"] = args.target_model
    results["draft_model"] = args.draft_model
    results["adapter_path"] = args.adapter_path
    results["domain"] = args.domain
    results["divergence_types"] = args.divergence_types
    results["max_seq_len"] = args.max_seq_len
    results["seed"] = args.seed

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
