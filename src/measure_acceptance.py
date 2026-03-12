"""Measure speculative decoding acceptance rate via manual simulation.

Simulates the speculative decoding verify-and-accept loop: the draft model
proposes K tokens greedily, the target model verifies them in a single forward
pass, and the longest matching prefix is accepted.  This gives fine-grained
metrics (per-position acceptance, mean accepted length) that are more
informative than wall-clock benchmarks alone.

Usage::

    python -m src.measure_acceptance \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --draft_model  meta-llama/Llama-3.1-1B-Instruct \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output results/exp1/eval_acceptance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("specaware.measure_acceptance")


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

def measure_acceptance_rate(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_prompts: list[str],
    target_device: torch.device,
    draft_device: torch.device,
    max_new_tokens: int = 128,
    draft_k: int = 5,
) -> dict:
    """Measure speculative decoding acceptance rate via manual simulation.

    For each prompt the draft model proposes ``draft_k`` tokens greedily and
    the target model verifies them in one forward pass.  The longest matching
    prefix (where target greedy == draft greedy) is accepted; remaining tokens
    are discarded and generation continues from the last accepted position.

    Args:
        target_model: The (possibly LoRA-adapted) target language model.
        draft_model: The frozen draft language model.
        tokenizer: Tokenizer shared by both models.
        eval_prompts: List of prompt strings to evaluate.
        target_device: Device the target model lives on.
        draft_device: Device the draft model lives on.
        max_new_tokens: Maximum number of new tokens to generate per prompt.
        draft_k: Number of tokens the draft model proposes per step.

    Returns:
        Dictionary containing:
            - ``alpha``: overall acceptance rate (total_accepted / total_drafted).
            - ``alpha_std``: standard deviation of per-prompt acceptance rates.
            - ``mean_accepted_length``: average consecutive tokens accepted per
              verification step.
            - ``per_position_alpha``: list of ``draft_k`` floats giving the
              acceptance rate at each draft position (0-indexed).
            - ``per_prompt_alpha``: list of per-prompt acceptance rates.
            - ``num_prompts``: number of prompts evaluated.
            - ``total_tokens_generated``: total tokens generated across prompts.
    """
    target_model.eval()
    draft_model.eval()

    total_drafted = 0
    total_accepted = 0
    per_prompt_alphas: list[float] = []
    # per_position_accepted[i] / per_position_total[i] gives positional alpha
    per_position_accepted = [0] * draft_k
    per_position_total = [0] * draft_k
    all_accepted_lengths: list[int] = []
    total_tokens_generated = 0

    for prompt_idx, prompt in enumerate(eval_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # Work on target device for the running sequence
        generated_ids = input_ids.to(target_device)

        prompt_drafted = 0
        prompt_accepted = 0
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            # ---------------------------------------------------------------
            # Step A: Draft model proposes K tokens greedily (autoregressive)
            # ---------------------------------------------------------------
            draft_input_ids = generated_ids.to(draft_device)
            drafted_tokens: list[int] = []

            with torch.no_grad():
                for _k in range(draft_k):
                    draft_out = draft_model(input_ids=draft_input_ids)
                    # Greedy: take argmax of last position
                    next_token = draft_out.logits[:, -1, :].argmax(dim=-1)  # (1,)
                    drafted_tokens.append(next_token.item())
                    # Append to draft input for next autoregressive step
                    draft_input_ids = torch.cat(
                        [draft_input_ids, next_token.unsqueeze(-1)], dim=-1
                    )

            # ---------------------------------------------------------------
            # Step B: Target model verifies all K tokens in one forward pass
            # ---------------------------------------------------------------
            # Build verification input: prompt + all drafted tokens
            drafted_tensor = torch.tensor(
                [drafted_tokens], device=target_device, dtype=generated_ids.dtype
            )
            verify_input = torch.cat([generated_ids, drafted_tensor], dim=-1)

            with torch.no_grad():
                target_out = target_model(input_ids=verify_input)

            # target_out.logits: (1, seq_len, vocab_size)
            # For position i in the original sequence, logits[0, i, :] predicts
            # the token at position i+1.  We need to check positions
            # [len(generated_ids)-1 .. len(generated_ids)-1+K-1] which predict
            # tokens at [len(generated_ids) .. len(generated_ids)+K-1], i.e.
            # the drafted tokens.
            prefix_len = generated_ids.size(1)

            # ---------------------------------------------------------------
            # Step C: Count accepted = longest matching prefix
            # ---------------------------------------------------------------
            num_accepted = 0
            for k_idx in range(draft_k):
                # logits at position (prefix_len - 1 + k_idx) predict the token
                # at position (prefix_len + k_idx), which is drafted_tokens[k_idx].
                verify_pos = prefix_len - 1 + k_idx
                target_greedy = target_out.logits[0, verify_pos, :].argmax(dim=-1).item()

                per_position_total[k_idx] += 1

                if target_greedy == drafted_tokens[k_idx]:
                    num_accepted += 1
                    per_position_accepted[k_idx] += 1
                else:
                    break

            prompt_drafted += draft_k
            prompt_accepted += num_accepted
            all_accepted_lengths.append(num_accepted)

            # ---------------------------------------------------------------
            # Step D: Keep accepted tokens, continue from last accepted pos
            # ---------------------------------------------------------------
            if num_accepted > 0:
                accepted_tensor = torch.tensor(
                    [drafted_tokens[:num_accepted]],
                    device=target_device,
                    dtype=generated_ids.dtype,
                )
                generated_ids = torch.cat([generated_ids, accepted_tensor], dim=-1)
                tokens_generated += num_accepted
            else:
                # No draft token accepted. Use target's own greedy token at
                # the first verification position instead, so generation makes
                # progress.
                target_greedy_token = target_out.logits[
                    0, prefix_len - 1, :
                ].argmax(dim=-1).item()
                fallback = torch.tensor(
                    [[target_greedy_token]],
                    device=target_device,
                    dtype=generated_ids.dtype,
                )
                generated_ids = torch.cat([generated_ids, fallback], dim=-1)
                tokens_generated += 1

            # Stop if EOS generated
            if generated_ids[0, -1].item() == tokenizer.eos_token_id:
                break

        # Per-prompt stats
        prompt_alpha = prompt_accepted / max(prompt_drafted, 1)
        per_prompt_alphas.append(prompt_alpha)
        total_drafted += prompt_drafted
        total_accepted += prompt_accepted
        total_tokens_generated += tokens_generated

        if (prompt_idx + 1) % 10 == 0 or (prompt_idx + 1) == len(eval_prompts):
            logger.info(
                "Prompt %d/%d  |  prompt_alpha=%.4f  |  running_alpha=%.4f",
                prompt_idx + 1,
                len(eval_prompts),
                prompt_alpha,
                total_accepted / max(total_drafted, 1),
            )

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    alpha = total_accepted / max(total_drafted, 1)
    alpha_std = float(torch.tensor(per_prompt_alphas).std().item()) if len(per_prompt_alphas) > 1 else 0.0
    mean_accepted_length = (
        sum(all_accepted_lengths) / max(len(all_accepted_lengths), 1)
    )
    per_position_alpha = [
        per_position_accepted[i] / max(per_position_total[i], 1)
        for i in range(draft_k)
    ]

    results = {
        "alpha": alpha,
        "alpha_std": alpha_std,
        "mean_accepted_length": mean_accepted_length,
        "per_position_alpha": per_position_alpha,
        "per_prompt_alpha": per_prompt_alphas,
        "num_prompts": len(eval_prompts),
        "total_tokens_generated": total_tokens_generated,
        "draft_k": draft_k,
        "max_new_tokens": max_new_tokens,
    }

    logger.info(
        "Final: alpha=%.4f +/- %.4f  |  mean_accepted_len=%.2f  |  "
        "total_drafted=%d  total_accepted=%d",
        alpha,
        alpha_std,
        mean_accepted_length,
        total_drafted,
        total_accepted,
    )
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

    The YAML file is expected to have top-level keys for each domain (e.g.
    ``code``, ``medical``, ``chat``, ``mixed``), each mapping to a list of
    prompt strings.

    Args:
        prompts_file: Path to the YAML prompts file.
        domain: If provided, load only prompts for this domain.  If ``None``,
            load all prompts concatenated.

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
    """Parse command-line arguments for acceptance rate measurement.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Measure speculative decoding acceptance rate",
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
        default="results/eval_acceptance.json",
        help="Path to write the output JSON results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--draft_k",
        type=int,
        default=5,
        help="Number of tokens the draft model proposes per step.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Limit to first N prompts (useful for quick tests). Default: use all.",
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
    """Entry point: load models, run acceptance measurement, save results."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Import here to avoid circular imports at module level
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
    if args.num_prompts is not None:
        prompts = prompts[:args.num_prompts]
        logger.info("Limited to first %d prompts", args.num_prompts)

    # Run measurement
    results = measure_acceptance_rate(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        eval_prompts=prompts,
        target_device=target_device,
        draft_device=draft_device,
        max_new_tokens=args.max_new_tokens,
        draft_k=args.draft_k,
    )

    # Add metadata
    results["target_model"] = args.target_model
    results["draft_model"] = args.draft_model
    results["adapter_path"] = args.adapter_path
    results["domain"] = args.domain
    results["seed"] = args.seed

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # Print summary
    logger.info("=" * 60)
    logger.info("ACCEPTANCE RATE SUMMARY")
    logger.info("=" * 60)
    logger.info("  alpha (overall):       %.4f +/- %.4f", results["alpha"], results["alpha_std"])
    logger.info("  mean accepted length:  %.2f / %d", results["mean_accepted_length"], args.draft_k)
    logger.info("  per-position alpha:    %s", [f"{a:.3f}" for a in results["per_position_alpha"]])
    logger.info("  num prompts:           %d", results["num_prompts"])
    logger.info("  total tokens generated: %d", results["total_tokens_generated"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
