"""Utility functions for speculator-aware fine-tuning.

Provides config loading, seed setting, logging setup, device detection,
checkpoint helpers, and CLI argument parsing.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Args:
        base: Base dictionary (will not be mutated).
        override: Dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _auto_cast(value: str) -> int | float | bool | str | None:
    """Attempt to cast a string value to int, float, bool, or None.

    Args:
        value: Raw string from CLI.

    Returns:
        The value cast to the most specific matching type.
    """
    if value.lower() == "null" or value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides in ``--key.subkey value`` format to config dict.

    Supports nested dot-notation keys. Values are auto-cast to int/float/bool
    where possible.

    Args:
        config: Config dictionary to update (mutated in-place).
        overrides: List of strings, e.g. ``["--spec_loss.lam", "0.05", "--domain", "code"]``.

    Returns:
        The updated config dictionary.
    """
    i = 0
    while i < len(overrides):
        token = overrides[i]
        if not token.startswith("--"):
            i += 1
            continue

        key_path = token.lstrip("-")
        if i + 1 >= len(overrides) or overrides[i + 1].startswith("--"):
            # Boolean flag with no value — treat as True
            value: Any = True
        else:
            value = _auto_cast(overrides[i + 1])
            i += 1

        # Walk into nested dicts
        keys = key_path.split(".")
        target = config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

        i += 1
    return config


def load_config(config_path: str, cli_overrides: list[str] | None = None) -> dict:
    """Load a YAML experiment config and merge with model defaults.

    The function first loads ``configs/models.yaml`` (resolved relative to the
    repository root) as a base, then deep-merges the experiment-specific config
    on top, and finally applies any CLI overrides.

    Args:
        config_path: Path to the experiment YAML config file.
        cli_overrides: Optional list of CLI override tokens, e.g.
            ``["--spec_loss.lam", "0.05"]``.

    Returns:
        Fully merged config dictionary.
    """
    config_path = os.path.abspath(config_path)

    # Resolve repo root: walk up from this file (src/utils.py) to repo root
    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "configs" / "models.yaml"

    # Load defaults if available
    defaults: dict = {}
    if defaults_path.exists():
        with open(defaults_path, "r") as f:
            defaults = yaml.safe_load(f) or {}

    # Load experiment config
    with open(config_path, "r") as f:
        experiment_config: dict = yaml.safe_load(f) or {}

    # Merge: experiment overrides defaults
    config = _deep_merge(defaults, experiment_config)

    # Apply CLI overrides
    if cli_overrides:
        _apply_overrides(config, cli_overrides)

    return config


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all relevant libraries.

    Sets seeds for ``random``, ``numpy``, ``torch`` (CPU and CUDA), and
    ``transformers`` (if installed).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set transformers seed if available
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str, name: str = "specaware") -> logging.Logger:
    """Configure Python logging to both console (INFO) and file (DEBUG).

    Creates ``output_dir`` if it does not exist and writes logs to
    ``output_dir/train.log``.

    Args:
        output_dir: Directory where the log file will be written.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — DEBUG level
    log_file = os.path.join(output_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_devices(
    target_device: str | None = None,
    draft_device: str | None = None,
) -> tuple[torch.device, torch.device]:
    """Detect available GPUs and assign devices for target and draft models.

    Auto-assignment logic (when arguments are None):
    - 2+ GPUs available: target -> cuda:0, draft -> cuda:1
    - 1 GPU available: both -> cuda:0
    - No GPU: both -> cpu

    Args:
        target_device: Explicit device string for the target model (e.g. ``"cuda:0"``).
        draft_device: Explicit device string for the draft model (e.g. ``"cuda:1"``).

    Returns:
        Tuple of ``(target_device, draft_device)`` as ``torch.device`` objects.
    """
    num_gpus = torch.cuda.device_count()

    if target_device is not None:
        t_dev = torch.device(target_device)
    elif num_gpus >= 1:
        t_dev = torch.device("cuda:0")
    else:
        t_dev = torch.device("cpu")

    if draft_device is not None:
        d_dev = torch.device(draft_device)
    elif num_gpus >= 2:
        d_dev = torch.device("cuda:1")
    elif num_gpus == 1:
        d_dev = torch.device("cuda:0")
    else:
        d_dev = torch.device("cpu")

    return t_dev, d_dev


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: Any, output_dir: str, step: int | None = None) -> str:
    """Save LoRA adapter weights using PEFT's ``save_pretrained``.

    If ``step`` is provided, saves to a subdirectory ``checkpoints/step_{step}``.
    Otherwise saves to ``final/`` within ``output_dir``.

    Args:
        model: A PEFT model with a ``save_pretrained`` method.
        output_dir: Base output directory for the experiment run.
        step: Optional training step number for intermediate checkpoints.

    Returns:
        Path to the directory where the adapter was saved.
    """
    if step is not None:
        save_dir = os.path.join(output_dir, "checkpoints", f"step_{step}")
    else:
        save_dir = os.path.join(output_dir, "final")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

    # Fallback: ensure adapter_config.json is saved (some PEFT versions skip it)
    config_path = os.path.join(save_dir, "adapter_config.json")
    if not os.path.exists(config_path) and hasattr(model, "peft_config"):
        import json
        active_adapter = model.active_adapter
        if isinstance(active_adapter, list):
            active_adapter = active_adapter[0]
        cfg = model.peft_config[active_adapter]
        with open(config_path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    logger = logging.getLogger("specaware")
    logger.info("Saved checkpoint to %s", save_dir)

    return save_dir


def save_metrics(
    metrics: list[dict],
    output_dir: str,
    filename: str = "training_metrics.json",
) -> None:
    """Save a list of metric dictionaries to a JSON file.

    Args:
        metrics: List of per-step metric dicts (e.g. containing ``step``,
            ``task_loss``, ``spec_loss``, ``acceptance_proxy``, ``lr``).
        output_dir: Directory where the JSON file will be written.
        filename: Name of the output JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger = logging.getLogger("specaware")
    logger.debug("Saved %d metric entries to %s", len(metrics), path)


def save_config(config: dict, output_dir: str) -> None:
    """Save the exact config used for an experiment run.

    Writes to ``output_dir/config.yaml`` so results are fully reproducible.

    Args:
        config: The complete config dictionary (with all overrides applied).
        output_dir: Directory where ``config.yaml`` will be written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger = logging.getLogger("specaware")
    logger.debug("Saved config to %s", path)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Recognized arguments:
        --config: Path to YAML config file (required).
        --target_device: Device for target model (e.g. ``cuda:0``).
        --draft_device: Device for draft model (e.g. ``cuda:1``).
        --domain: Override domain in config.

    Any additional ``--key.subkey value`` arguments are collected and passed
    through as config overrides via ``load_config``.

    Returns:
        Parsed namespace with ``config``, ``target_device``, ``draft_device``,
        ``domain``, and ``overrides`` (list of remaining tokens).
    """
    parser = argparse.ArgumentParser(
        description="Speculator-aware fine-tuning training script",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config file.",
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
        "--domain",
        type=str,
        default=None,
        help="Override domain in config (code, medical, chat).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory in config.",
    )

    args, remaining = parser.parse_known_args()
    args.overrides = remaining
    return args
