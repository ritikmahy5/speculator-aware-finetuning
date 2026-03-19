"""DPO preference dataset loading and preprocessing.

Loads preference pairs (chosen/rejected completions) for Direct Preference
Optimization training.  Supports HuggingFaceH4/ultrafeedback_binarized
(primary) and Anthropic/hh-rlhf (fallback).

The loader returns tokenized preference pairs with prompt/completion labels
properly masked so that the DPO loss is only computed over completion tokens.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 50


def _extract_ultrafeedback(
    sample: dict,
    tokenizer,
    max_seq_len: int,
) -> Optional[dict]:
    """Extract and tokenize a preference pair from ultrafeedback_binarized.

    Args:
        sample: Raw sample with ``chosen`` and ``rejected`` message lists.
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        max_seq_len: Maximum sequence length for tokenization.

    Returns:
        Dict with tokenized fields, or None if the sample is invalid.
    """
    chosen_messages = sample.get("chosen")
    rejected_messages = sample.get("rejected")

    if not chosen_messages or not rejected_messages:
        return None
    if not isinstance(chosen_messages, list) or not isinstance(rejected_messages, list):
        return None

    # Separate prompt from completion
    # In ultrafeedback, the last message is the assistant response
    if len(chosen_messages) < 2 or len(rejected_messages) < 2:
        return None

    # The prompt is all messages except the last (assistant) turn
    prompt_messages = chosen_messages[:-1]

    try:
        # Tokenize prompt only (for label masking)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        chosen_text = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False,
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        return None

    if len(chosen_text.strip()) < MIN_TEXT_LENGTH:
        return None
    if len(rejected_text.strip()) < MIN_TEXT_LENGTH:
        return None

    # Tokenize prompt to find prompt length (for label masking)
    prompt_enc = tokenizer(
        prompt_text,
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_len = len(prompt_enc["input_ids"])

    # Tokenize full chosen and rejected sequences
    chosen_enc = tokenizer(
        chosen_text,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    rejected_enc = tokenizer(
        rejected_text,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Create labels: -100 for prompt tokens and padding, real ids for completion
    chosen_labels = chosen_enc["input_ids"].clone().squeeze(0)
    chosen_labels[:prompt_len] = -100
    chosen_labels[chosen_enc["attention_mask"].squeeze(0) == 0] = -100

    rejected_labels = rejected_enc["input_ids"].clone().squeeze(0)
    rejected_labels[:prompt_len] = -100
    rejected_labels[rejected_enc["attention_mask"].squeeze(0) == 0] = -100

    return {
        "chosen_input_ids": chosen_enc["input_ids"].squeeze(0).tolist(),
        "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0).tolist(),
        "chosen_labels": chosen_labels.tolist(),
        "rejected_input_ids": rejected_enc["input_ids"].squeeze(0).tolist(),
        "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0).tolist(),
        "rejected_labels": rejected_labels.tolist(),
    }


def _extract_hh_rlhf(
    sample: dict,
    tokenizer,
    max_seq_len: int,
) -> Optional[dict]:
    """Extract and tokenize a preference pair from Anthropic/hh-rlhf.

    Args:
        sample: Raw sample with ``chosen`` and ``rejected`` text strings.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: Maximum sequence length.

    Returns:
        Dict with tokenized fields, or None if the sample is invalid.
    """
    chosen_text = sample.get("chosen", "")
    rejected_text = sample.get("rejected", "")

    if len(chosen_text.strip()) < MIN_TEXT_LENGTH:
        return None
    if len(rejected_text.strip()) < MIN_TEXT_LENGTH:
        return None

    # Find the last "Assistant:" to split prompt from completion
    # hh-rlhf format: "Human: ...\n\nAssistant: ..."
    def _find_prompt_len(text: str) -> int:
        last_assistant = text.rfind("Assistant:")
        if last_assistant == -1:
            return 0
        prompt_text = text[: last_assistant + len("Assistant: ")]
        enc = tokenizer(prompt_text, add_special_tokens=False)
        return len(enc["input_ids"])

    chosen_prompt_len = _find_prompt_len(chosen_text)
    rejected_prompt_len = _find_prompt_len(rejected_text)

    chosen_enc = tokenizer(
        chosen_text,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    rejected_enc = tokenizer(
        rejected_text,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    chosen_labels = chosen_enc["input_ids"].clone().squeeze(0)
    chosen_labels[:chosen_prompt_len] = -100
    chosen_labels[chosen_enc["attention_mask"].squeeze(0) == 0] = -100

    rejected_labels = rejected_enc["input_ids"].clone().squeeze(0)
    rejected_labels[:rejected_prompt_len] = -100
    rejected_labels[rejected_enc["attention_mask"].squeeze(0) == 0] = -100

    return {
        "chosen_input_ids": chosen_enc["input_ids"].squeeze(0).tolist(),
        "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0).tolist(),
        "chosen_labels": chosen_labels.tolist(),
        "rejected_input_ids": rejected_enc["input_ids"].squeeze(0).tolist(),
        "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0).tolist(),
        "rejected_labels": rejected_labels.tolist(),
    }


DPO_DATASET_CONFIGS = {
    "HuggingFaceH4/ultrafeedback_binarized": {
        "split": "train_prefs",
        "extractor": _extract_ultrafeedback,
    },
    "Anthropic/hh-rlhf": {
        "split": "train",
        "extractor": _extract_hh_rlhf,
    },
}


def load_dpo_dataset(
    dataset_name: str,
    tokenizer,
    max_samples: int = 10000,
    max_seq_len: int = 1024,
    seed: int = 42,
) -> Dataset:
    """Load and preprocess a DPO preference dataset.

    Streams the dataset, extracts and tokenizes preference pairs, and returns
    a HuggingFace Dataset with chosen/rejected input_ids, attention_masks,
    and labels (prompt tokens masked to -100).

    Args:
        dataset_name: HuggingFace dataset identifier. Supported:
            ``HuggingFaceH4/ultrafeedback_binarized`` (primary),
            ``Anthropic/hh-rlhf`` (fallback).
        tokenizer: HuggingFace tokenizer instance.
        max_samples: Maximum number of preference pairs to load.
        max_seq_len: Maximum sequence length for tokenization.
        seed: Random seed for reproducibility.

    Returns:
        HuggingFace Dataset with columns: chosen_input_ids, chosen_attention_mask,
        chosen_labels, rejected_input_ids, rejected_attention_mask, rejected_labels.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    if dataset_name not in DPO_DATASET_CONFIGS:
        raise ValueError(
            f"Unknown DPO dataset: {dataset_name!r}. "
            f"Must be one of: {list(DPO_DATASET_CONFIGS.keys())}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token (%s)", tokenizer.eos_token)

    ds_config = DPO_DATASET_CONFIGS[dataset_name]
    extract_fn = ds_config["extractor"]

    logger.info(
        "Loading DPO dataset: %s (split=%s, max_samples=%d)",
        dataset_name,
        ds_config["split"],
        max_samples,
    )

    ds = load_dataset(dataset_name, split=ds_config["split"], streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    samples: list[dict] = []
    skipped = 0

    for raw_sample in ds:
        if len(samples) >= max_samples:
            break

        processed = extract_fn(raw_sample, tokenizer, max_seq_len)
        if processed is None:
            skipped += 1
            continue

        samples.append(processed)

    if not samples:
        raise RuntimeError(f"No valid preference pairs collected from {dataset_name}")

    logger.info(
        "Collected %d preference pairs (skipped %d invalid samples)",
        len(samples),
        skipped,
    )

    # Build dataset from list of dicts
    dataset = Dataset.from_dict({
        key: [s[key] for s in samples]
        for key in samples[0].keys()
    })
    dataset = dataset.shuffle(seed=seed)
    dataset.set_format(
        type="torch",
        columns=[
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ],
    )

    logger.info("DPO dataset ready: %d pairs, max_seq_len=%d", len(dataset), max_seq_len)
    return dataset


def create_dpo_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a PyTorch DataLoader for DPO preference data.

    Args:
        dataset: Preprocessed DPO dataset from ``load_dpo_dataset``.
        batch_size: Number of preference pairs per batch.
        shuffle: Whether to shuffle data each epoch.
        num_workers: Number of data loading subprocesses.

    Returns:
        DataLoader yielding batches of dicts with chosen_* and rejected_* tensors.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(
        "DPO DataLoader: batch_size=%d, num_batches=%d",
        batch_size,
        len(dataloader),
    )
    return dataloader
