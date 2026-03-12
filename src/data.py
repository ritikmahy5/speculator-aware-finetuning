"""Dataset loading and preprocessing for speculator-aware fine-tuning.

Supports three domains: code, medical, and chat. Each domain has its own
dataset source and text extraction logic. Datasets are loaded via HuggingFace
with streaming=True to avoid downloading full datasets upfront.
"""

import logging
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Dataset configurations per domain
DOMAIN_CONFIGS = {
    "code": {
        "path": "iamtarun/python_code_instructions_18k_alpaca",
        "data_dir": None,
        "split": "train",
    },
    "medical": {
        "path": "medalpaca/medical_meadow_medqa",
        "data_dir": None,
        "split": "train",
    },
    "chat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "data_dir": None,
        "split": "train_sft",
    },
}

MIN_TEXT_LENGTH = 50


def get_domain_text(sample: dict, domain: str, tokenizer=None) -> Optional[str]:
    """Extract text from a raw sample based on domain.

    Args:
        sample: A single raw sample dict from the streaming dataset.
        domain: One of "code", "medical", or "chat".
        tokenizer: Required for the "chat" domain to apply the chat template.
            Ignored for other domains.

    Returns:
        The extracted text string, or None if the text is too short (< 50 chars)
        or if required fields are missing.
    """
    text = None

    if domain == "code":
        # Use instruction + code output for a complete coding sample
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        if not output:
            # Fallback for datasets with a "content" field (e.g. starcoderdata)
            text = sample.get("content")
            if not text or not isinstance(text, str):
                return None
        else:
            parts = []
            if instruction:
                parts.append(f"# Task: {instruction}")
            parts.append(output)
            text = "\n".join(parts)

    elif domain == "medical":
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        if not input_text and not output_text:
            return None
        # Format as question-answer pair
        parts = []
        if input_text:
            parts.append(f"Question: {input_text}")
        if output_text:
            parts.append(f"Answer: {output_text}")
        text = "\n".join(parts)

    elif domain == "chat":
        messages = sample.get("messages")
        if not messages or not isinstance(messages, list):
            return None
        if tokenizer is None:
            logger.warning("Tokenizer required for chat domain but not provided.")
            return None
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            logger.debug("Failed to apply chat template: %s", e)
            return None

    else:
        raise ValueError(f"Unknown domain: {domain!r}. Must be one of: code, medical, chat")

    if text is None or len(text.strip()) < MIN_TEXT_LENGTH:
        return None

    return text


def _collect_streaming_samples(
    domain: str,
    tokenizer,
    max_samples: int,
    seed: int,
) -> list[str]:
    """Iterate over a streaming dataset and collect up to max_samples texts.

    Args:
        domain: One of "code", "medical", or "chat".
        tokenizer: Tokenizer instance (needed for chat template).
        max_samples: Maximum number of samples to collect.
        seed: Random seed for shuffling the streaming buffer.

    Returns:
        A list of extracted text strings.
    """
    config = DOMAIN_CONFIGS[domain]

    load_kwargs = {
        "path": config["path"],
        "split": config["split"],
        "streaming": True,
    }
    if config["data_dir"] is not None:
        load_kwargs["data_dir"] = config["data_dir"]

    logger.info(
        "Loading streaming dataset for domain=%s from %s (split=%s)",
        domain,
        config["path"],
        config["split"],
    )

    ds = load_dataset(**load_kwargs)
    # Shuffle the streaming dataset buffer for some randomness
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    texts: list[str] = []
    skipped = 0

    for sample in ds:
        if len(texts) >= max_samples:
            break

        text = get_domain_text(sample, domain, tokenizer=tokenizer)
        if text is None:
            skipped += 1
            continue

        texts.append(text)

    logger.info(
        "Collected %d samples for domain=%s (skipped %d short/invalid samples)",
        len(texts),
        domain,
        skipped,
    )
    return texts


def load_dataset_for_domain(
    domain: str,
    tokenizer,
    max_samples: int = 10000,
    max_seq_len: int = 1024,
    seed: int = 42,
) -> Dataset:
    """Load and preprocess a dataset for a given domain.

    Streams the raw dataset, extracts text per domain, tokenizes, and returns
    a HuggingFace Dataset ready for training. Labels are set to input_ids with
    padding positions masked to -100 for causal LM loss.

    Args:
        domain: One of "code", "medical", or "chat".
        tokenizer: A HuggingFace tokenizer. If it lacks a pad_token, eos_token
            will be used as the pad token.
        max_samples: Maximum number of samples to include.
        max_seq_len: Maximum sequence length for tokenization (truncation and
            padding target).
        seed: Random seed for reproducibility (shuffling).

    Returns:
        A HuggingFace Dataset with columns: input_ids, attention_mask, labels.
        Each value is a list of ints (not tensors) suitable for DataLoader
        collation.

    Raises:
        ValueError: If domain is not recognized.
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(
            f"Unknown domain: {domain!r}. Must be one of: {list(DOMAIN_CONFIGS.keys())}"
        )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token (%s)", tokenizer.eos_token)

    # Collect raw texts from the streaming dataset
    texts = _collect_streaming_samples(domain, tokenizer, max_samples, seed)

    if not texts:
        raise RuntimeError(f"No valid samples collected for domain={domain!r}")

    # Tokenize all texts in batch
    logger.info("Tokenizing %d samples (max_seq_len=%d)...", len(texts), max_seq_len)
    encodings = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Labels = input_ids, with padding positions set to -100 so they are
    # ignored by CrossEntropyLoss
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Build a HuggingFace Dataset from the tensors (store as lists for
    # compatibility with Dataset and default collation)
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist(),
        }
    )

    # Shuffle the final dataset with the seed
    dataset = dataset.shuffle(seed=seed)

    # Set the format so __getitem__ returns torch tensors
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info(
        "Dataset ready: %d samples, seq_len=%d, domain=%s",
        len(dataset),
        max_seq_len,
        domain,
    )
    return dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a PyTorch DataLoader from a preprocessed dataset.

    Args:
        dataset: A HuggingFace Dataset with torch format set (columns:
            input_ids, attention_mask, labels).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        num_workers: Number of subprocesses for data loading. Set to 0 for
            debugging.

    Returns:
        A PyTorch DataLoader yielding batches of dicts with keys:
        input_ids, attention_mask, labels (each a torch.Tensor).
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
        "DataLoader created: batch_size=%d, shuffle=%s, num_batches=%d",
        batch_size,
        shuffle,
        len(dataloader),
    )
    return dataloader
