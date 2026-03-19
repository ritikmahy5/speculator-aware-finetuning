"""Evaluate task performance of fine-tuned models.

Measures downstream task quality to quantify the task-α tradeoff:
- Code domain: HumanEval pass@1
- Medical domain: MedQA accuracy (USMLE-style multiple choice)
- Chat domain: Perplexity on held-out UltraChat samples

Uses the lm-eval harness for HumanEval and MedQA. Falls back to perplexity
measurement if lm-eval tasks are unavailable.

Usage::

    python -m src.eval_task \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path results/exp4_llama/code_lam_0.1/final \
        --tasks humaneval,medqa \
        --output results/eval_task.json

    python -m src.eval_task \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path results/exp4_llama/chat_lam_0.5/final \
        --tasks perplexity \
        --domain chat \
        --output results/eval_task_chat.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("specaware.eval_task")


# ---------------------------------------------------------------------------
# Perplexity measurement on held-out data
# ---------------------------------------------------------------------------

def measure_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 1024,
) -> dict:
    """Measure perplexity on a list of text samples.

    Args:
        model: The language model to evaluate.
        tokenizer: Tokenizer for the model.
        texts: List of text strings to compute perplexity on.
        device: Device the model is on.
        max_seq_len: Maximum sequence length for tokenization.

    Returns:
        Dictionary with perplexity stats.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    per_sample_ppl = []

    with torch.no_grad():
        for idx, text in enumerate(texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
            n_tokens = input_ids.size(1) - 1  # shifted labels

            total_loss += loss * n_tokens
            total_tokens += n_tokens
            per_sample_ppl.append(math.exp(loss))

            if (idx + 1) % 50 == 0 or (idx + 1) == len(texts):
                logger.info(
                    "Sample %d/%d | ppl=%.2f | running_ppl=%.2f",
                    idx + 1, len(texts), per_sample_ppl[-1],
                    math.exp(total_loss / max(total_tokens, 1)),
                )

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    ppl_std = float(torch.tensor(per_sample_ppl).std().item()) if len(per_sample_ppl) > 1 else 0.0

    return {
        "perplexity": perplexity,
        "perplexity_std": ppl_std,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(per_sample_ppl),
        "per_sample_perplexity": per_sample_ppl,
    }


# ---------------------------------------------------------------------------
# lm-eval harness wrapper
# ---------------------------------------------------------------------------

def run_lm_eval(
    model_name: str,
    adapter_path: Optional[str],
    tasks: list[str],
    device: str = "cuda:0",
    num_fewshot: Optional[int] = None,
    batch_size: int = 4,
) -> dict:
    """Run lm-eval harness on specified tasks.

    Args:
        model_name: HuggingFace model ID.
        adapter_path: Optional path to PEFT LoRA adapter.
        tasks: List of lm-eval task names.
        device: Device string.
        num_fewshot: Number of few-shot examples (None = task default).
        batch_size: Evaluation batch size.

    Returns:
        Dictionary with results per task.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
        return {"error": "lm-eval not installed"}

    logger.info("Running lm-eval: tasks=%s, model=%s, adapter=%s", tasks, model_name, adapter_path)

    # Build model arguments for lm-eval
    model_args = f"pretrained={model_name},dtype=bfloat16,device_map_option=auto"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    lm = HFLM(
        pretrained=model_name,
        dtype="bfloat16",
        device=device,
        peft=adapter_path,
        batch_size=batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        confirm_run_unsafe_code=True,
    )

    # Extract key metrics
    output = {}
    for task_name, task_results in results.get("results", {}).items():
        output[task_name] = {}
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                output[task_name][metric] = value
            elif metric.endswith(",none"):
                clean_metric = metric.replace(",none", "")
                output[task_name][clean_metric] = value

    logger.info("lm-eval results: %s", json.dumps(output, indent=2))
    return output


# ---------------------------------------------------------------------------
# Domain-specific held-out data loading
# ---------------------------------------------------------------------------

def load_held_out_data(domain: str, num_samples: int = 200, prompts_file: str = "configs/eval_prompts.yaml") -> list[str]:
    """Load held-out data for perplexity evaluation.

    First tries to load from the eval prompts file (always available).
    Falls back to streaming datasets if more samples are needed.

    Args:
        domain: One of "code", "medical", "chat".
        num_samples: Number of samples to load.
        prompts_file: Path to eval prompts YAML.

    Returns:
        List of text strings.
    """
    texts = []

    # Primary: use eval prompts (guaranteed available, no gating issues)
    if os.path.exists(prompts_file):
        with open(prompts_file, "r") as f:
            prompts_data = yaml.safe_load(f)
        key = domain if domain in prompts_data else f"{domain}_prompts"
        if key in prompts_data:
            texts = prompts_data[key]
            # Also add mixed prompts for more data
            mixed_key = "mixed" if "mixed" in prompts_data else "mixed_prompts"
            if mixed_key in prompts_data:
                texts = texts + prompts_data[mixed_key]
            logger.info("Loaded %d prompts from %s for domain '%s'", len(texts), prompts_file, domain)

    # Fallback: try streaming datasets for more samples
    if len(texts) < num_samples:
        try:
            from datasets import load_dataset as _load_ds
            logger.info("Loading additional samples from streaming dataset...")

            if domain == "code":
                ds = _load_ds("codeparrot/github-code-clean", split="train", streaming=True, trust_remote_code=True)
                for i, sample in enumerate(ds):
                    if len(texts) >= num_samples:
                        break
                    content = sample.get("code", sample.get("content", ""))
                    if len(content) >= 50:
                        texts.append(content)
            elif domain == "medical":
                ds = _load_ds("medalpaca/medical_meadow_medqa", split="train", streaming=True)
                for i, sample in enumerate(ds):
                    if i < 15000 or len(texts) >= num_samples:
                        if len(texts) >= num_samples:
                            break
                        continue
                    text = f"Question: {sample.get('input', '')}\nAnswer: {sample.get('output', '')}"
                    if len(text) >= 50:
                        texts.append(text)
            elif domain == "chat":
                ds = _load_ds("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
                for i, sample in enumerate(ds):
                    if i < 15000 or len(texts) >= num_samples:
                        if len(texts) >= num_samples:
                            break
                        continue
                    messages = sample.get("messages", [])
                    text = "\n".join(m.get("content", "") for m in messages)
                    if len(text) >= 50:
                        texts.append(text)
        except Exception as e:
            logger.warning("Could not load streaming dataset: %s. Using eval prompts only.", e)

    texts = texts[:num_samples]
    logger.info("Final: %d samples for domain '%s'", len(texts), domain)
    return texts


# ---------------------------------------------------------------------------
# Model loading (same pattern as other measurement scripts)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    adapter_path: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer, optionally with a LoRA adapter."""
    logger.info("Loading model %s onto %s", model_name, device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device},
    )
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task performance of fine-tuned models")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument(
        "--tasks", type=str, default="perplexity",
        help="Comma-separated list of tasks: perplexity, humaneval, medqa_4options",
    )
    parser.add_argument("--domain", type=str, default="code", help="Domain for perplexity eval")
    parser.add_argument("--output", type=str, default="results/eval_task.json")
    parser.add_argument("--num_samples", type=int, default=200, help="Samples for perplexity eval")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    torch.manual_seed(args.seed)

    device = torch.device(args.target_device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    tasks = [t.strip() for t in args.tasks.split(",")]
    all_results: dict = {}

    # Perplexity evaluation
    if "perplexity" in tasks:
        logger.info("Running perplexity evaluation on domain=%s", args.domain)
        model, tokenizer = _load_model_and_tokenizer(args.target_model, device, args.adapter_path)
        texts = load_held_out_data(args.domain, args.num_samples)
        ppl_results = measure_perplexity(model, tokenizer, texts, device)
        all_results["perplexity"] = ppl_results
        # Free memory before lm-eval
        del model
        torch.cuda.empty_cache()

    # lm-eval tasks
    lm_eval_tasks = [t for t in tasks if t != "perplexity"]
    if lm_eval_tasks:
        lm_results = run_lm_eval(
            args.target_model,
            args.adapter_path,
            lm_eval_tasks,
            device=str(device),
            batch_size=args.batch_size,
        )
        all_results["lm_eval"] = lm_results

    # Metadata
    all_results["target_model"] = args.target_model
    all_results["adapter_path"] = args.adapter_path
    all_results["tasks"] = tasks
    all_results["domain"] = args.domain
    all_results["seed"] = args.seed

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
