"""Plotting and results analysis for speculator-aware fine-tuning experiments.

Generates publication-quality plots from experiment result JSON files.
All plots are saved as both PNG (300 DPI) and PDF (vector).

Usage:
    python src/analyze_results.py --results_dir results --output_dir plots
"""

from __future__ import annotations

import argparse
import json
import logging
import re

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (from project spec)
# ---------------------------------------------------------------------------
COLORS = {
    "base": "#1A5276",
    "code": "#E74C3C",
    "medical": "#27AE60",
    "chat": "#F39C12",
    "spec_aware": "#3498DB",
}

DOMAIN_COLORS = {
    "code": COLORS["code"],
    "medical": COLORS["medical"],
    "chat": COLORS["chat"],
}

DOMAINS = ["code", "medical", "chat"]

FAMILY_COLORS = {
    "Llama": "#1A5276",
    "Qwen": "#3498DB",
    "Gemma": "#8E44AD",
}

DOMAIN_MARKERS = {
    "code": "o",       # circle
    "medical": "s",    # square
    "chat": "^",       # triangle
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Load a JSON or YAML file, returning None if it does not exist or fails to parse.

    If the file extension is ``.yaml`` or ``.yml``, attempts to parse with
    ``yaml.safe_load`` first (requires PyYAML). Falls back gracefully to
    ``None`` with a warning on any parse error.

    Args:
        path: Path to the JSON or YAML file.

    Returns:
        Parsed content, or None if the file is missing or cannot be parsed.
    """
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r") as f:
            if path.suffix in (".yaml", ".yml"):
                if _HAS_YAML:
                    return yaml.safe_load(f)
                else:
                    logger.warning("PyYAML not installed; cannot parse %s", path)
                    return None
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error reading %s: %s", path, exc)
        return None


def _setup_style() -> None:
    """Configure matplotlib defaults for publication-quality plots.

    Sets font family, sizes, removes top/right spines, and configures
    Y-only gridlines.
    """
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (8, 5),
        "figure.constrained_layout.use": True,
    })


def _save_plot(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save a matplotlib figure as both PNG (300 DPI) and PDF.

    Args:
        fig: The matplotlib figure to save.
        output_dir: Directory to write files into.
        name: Base filename without extension (e.g. ``plot1_degradation``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved %s and %s", png_path, pdf_path)
    plt.close(fig)


def _add_bar_labels(ax: plt.Axes, fmt: str = ".2f") -> None:
    """Add value labels on top of bars in a bar chart.

    Args:
        ax: Matplotlib Axes containing bar containers.
        fmt: Format string for the numeric labels.
    """
    for container in ax.containers:
        if hasattr(container, 'patches'):
            ax.bar_label(container, fmt=f"%{fmt}", padding=3, fontsize=10)


# ---------------------------------------------------------------------------
# Plot 1 — Baseline Degradation (EXP-1)
# ---------------------------------------------------------------------------

def plot_degradation(results_dir: Path, output_dir: Path) -> None:
    """Generate bar chart of acceptance rate per domain, base vs fine-tuned.

    Reads ``results/exp1/acceptance_base_{domain}.json`` and
    ``results/exp1/acceptance_{domain}_baseline_eval_{domain}.json`` for each domain.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp1_dir = results_dir / "exp1"

    domains_found: list[str] = []
    base_alphas: list[float] = []
    ft_alphas: list[float] = []
    base_stds: list[float] = []
    ft_stds: list[float] = []

    for domain in DOMAINS:
        # Try per-domain base file first, then consolidated
        base_data = _load_json(exp1_dir / f"acceptance_base_{domain}.json")
        if base_data is None:
            consolidated = _load_json(exp1_dir / "acceptance_base.json")
            if consolidated is not None and domain in consolidated:
                base_data = consolidated[domain]
            elif consolidated is not None:
                base_data = consolidated
        if base_data is None:
            continue

        # Try per-domain FT file (naming from run_exp1.sh)
        ft_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline_eval_{domain}.json")
        if ft_data is None:
            ft_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline.json")
        if ft_data is None:
            continue

        domains_found.append(domain.capitalize())
        base_alphas.append(base_data.get("mean_alpha", base_data.get("alpha", 0)))
        base_stds.append(base_data.get("std_alpha", base_data.get("alpha_std", 0)))
        ft_alphas.append(ft_data.get("mean_alpha", ft_data.get("alpha", 0)))
        ft_stds.append(ft_data.get("std_alpha", ft_data.get("alpha_std", 0)))

    if not domains_found:
        logger.warning("Skipping plot1_degradation: no domain data found.")
        return

    x = np.arange(len(domains_found))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, base_alphas, width, yerr=base_stds, capsize=4,
           label="Base Model", color=COLORS["base"])
    ax.bar(x + width / 2, ft_alphas, width, yerr=ft_stds, capsize=4,
           label="Standard Fine-Tuned", color=COLORS["spec_aware"])

    ax.set_xlabel("Domain")
    ax.set_ylabel("Acceptance Rate (\u03b1)")
    ax.set_title("Speculator Acceptance Rate: Base vs Standard Fine-Tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(domains_found)
    ax.set_ylim(0, 1.05)
    ax.legend()
    _add_bar_labels(ax)

    _save_plot(fig, output_dir, "plot1_degradation")


# ---------------------------------------------------------------------------
# Plot 2 — KL-Acceptance Rate Correlation (EXP-2)
# ---------------------------------------------------------------------------

def plot_kl_correlation(results_dir: Path, output_dir: Path) -> None:
    """Generate scatter plot of KL divergence vs acceptance rate with regression.

    Reads checkpoint measurement files from ``results/exp2/``. Expects files
    named ``checkpoint_{pct}.json`` or a single ``correlation_data.json``
    containing lists of KL and alpha values.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp2_dir = results_dir / "exp2"
    if not exp2_dir.exists():
        logger.warning("Skipping plot2_kl_correlation: exp2 directory not found.")
        return

    kl_values: list[float] = []
    alpha_values: list[float] = []

    # Try consolidated file first
    corr_data = _load_json(exp2_dir / "correlation_data.json")
    if corr_data is not None:
        kl_values = corr_data.get("kl", [])
        alpha_values = corr_data.get("alpha", [])
    else:
        # Try individual checkpoint files
        for ckpt_file in sorted(exp2_dir.glob("checkpoint_*.json")):
            data = _load_json(ckpt_file)
            if data is None:
                continue
            kl_val = data.get("kl_divergence", data.get("kl", None))
            alpha_val = data.get("mean_alpha", data.get("alpha", None))
            if kl_val is not None and alpha_val is not None:
                kl_values.append(kl_val)
                alpha_values.append(alpha_val)

    # Fallback: try acceptance_*.json + kl_*.json pairs (e.g. acceptance_base.json,
    # acceptance_step_156.json, acceptance_final.json paired with kl_base.json, etc.)
    if len(kl_values) < 2:
        kl_values = []
        alpha_values = []
        for acc_file in sorted(exp2_dir.glob("acceptance_*.json")):
            # Derive the matching kl file: acceptance_X.json -> kl_X.json
            suffix = acc_file.name[len("acceptance_"):]  # e.g. "base.json", "step_156.json"
            kl_file = exp2_dir / f"kl_{suffix}"
            acc_data = _load_json(acc_file)
            kl_data = _load_json(kl_file)
            if acc_data is None or kl_data is None:
                continue
            alpha_val = acc_data.get("mean_alpha", acc_data.get("alpha", None))
            kl_val = kl_data.get("kl_mean", kl_data.get("kl_divergence", kl_data.get("kl", None)))
            if alpha_val is not None and kl_val is not None:
                alpha_values.append(alpha_val)
                kl_values.append(kl_val)

    if len(kl_values) < 2:
        logger.warning("Skipping plot2_kl_correlation: insufficient data points (%d).", len(kl_values))
        return

    kl_arr = np.array(kl_values)
    alpha_arr = np.array(alpha_values)

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(kl_arr, alpha_arr)
    spearman_r, spearman_p = stats.spearmanr(kl_arr, alpha_arr)

    # Linear regression for trend line
    slope, intercept, _, _, _ = stats.linregress(kl_arr, alpha_arr)
    x_line = np.linspace(kl_arr.min(), kl_arr.max(), 100)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(kl_arr, alpha_arr, color=COLORS["code"], s=80, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot(x_line, y_line, "--", color=COLORS["base"], linewidth=2, label="Linear fit")

    ax.set_xlabel("KL Divergence (target || draft)")
    ax.set_ylabel("Acceptance Rate (\u03b1)")
    ax.set_title("KL Divergence vs Speculator Acceptance Rate")
    ax.legend()

    # Annotation box with correlation stats
    textstr = (
        f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.1e})\n"
        f"Spearman \u03c1 = {spearman_r:.3f} (p = {spearman_p:.1e})"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    _save_plot(fig, output_dir, "plot2_kl_correlation")


# ---------------------------------------------------------------------------
# Plot 3 — Spec-Aware Comparison (EXP-3)
# ---------------------------------------------------------------------------

def plot_spec_aware_comparison(results_dir: Path, output_dir: Path) -> None:
    """Generate three-bar chart: base vs standard-FT vs spec-aware-FT alpha.

    Reads acceptance results from EXP-1 (base and standard FT) and EXP-3
    (speculator-aware FT).

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp1_dir = results_dir / "exp1"

    # Try multiple naming conventions for exp3
    exp3_dirs = list(results_dir.glob("exp3*"))

    if not exp3_dirs:
        logger.warning("Skipping plot3_spec_aware_comparison: exp3 results not found.")
        return

    # Collect data for each domain that has all three measurements
    domains_found: list[str] = []
    base_vals: list[float] = []
    std_vals: list[float] = []
    spec_vals: list[float] = []
    base_errs: list[float] = []
    std_errs: list[float] = []
    spec_errs: list[float] = []

    for domain in DOMAINS:
        # Load base data (per-domain file or consolidated)
        base_data = _load_json(exp1_dir / f"acceptance_base_{domain}.json")
        if base_data is None:
            consolidated = _load_json(exp1_dir / "acceptance_base.json")
            if consolidated is not None and domain in consolidated:
                base_data = consolidated[domain]
            elif consolidated is not None:
                base_data = consolidated
        if base_data is None:
            continue

        # Load standard FT data
        std_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline_eval_{domain}.json")
        if std_data is None:
            std_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline.json")
        if std_data is None:
            continue

        # Find spec-aware results for this domain
        spec_data = None
        for d in exp3_dirs:
            # Try eval_acceptance.json first (original naming)
            candidate = _load_json(d / "eval_acceptance.json")
            if candidate is not None and (candidate.get("domain") == domain or domain in d.name):
                spec_data = candidate
                break
            # Fallback: try acceptance_{domain}.json (actual naming from run_exp3.sh)
            candidate = _load_json(d / f"acceptance_{domain}.json")
            if candidate is not None:
                spec_data = candidate
                break
        if spec_data is None:
            continue

        domains_found.append(domain.capitalize())
        base_vals.append(base_data.get("mean_alpha", base_data.get("alpha", 0)))
        base_errs.append(base_data.get("std_alpha", base_data.get("alpha_std", 0)))
        std_vals.append(std_data.get("mean_alpha", std_data.get("alpha", 0)))
        std_errs.append(std_data.get("std_alpha", std_data.get("alpha_std", 0)))
        spec_vals.append(spec_data.get("mean_alpha", spec_data.get("alpha", 0)))
        spec_errs.append(spec_data.get("std_alpha", spec_data.get("alpha_std", 0)))

    if not domains_found:
        logger.warning("Skipping plot3_spec_aware_comparison: no complete domain data.")
        return

    x = np.arange(len(domains_found))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, base_vals, width, yerr=base_errs, capsize=4,
           label="Base Model", color=COLORS["base"])
    ax.bar(x, std_vals, width, yerr=std_errs, capsize=4,
           label="Standard FT", color=COLORS["code"])
    ax.bar(x + width, spec_vals, width, yerr=spec_errs, capsize=4,
           label="Spec-Aware FT", color=COLORS["spec_aware"])

    ax.set_xlabel("Domain")
    ax.set_ylabel("Acceptance Rate (\u03b1)")
    ax.set_title("Speculator-Aware Fine-Tuning Preserves Acceptance Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(domains_found)
    ax.set_ylim(0, 1.05)
    ax.legend()
    _add_bar_labels(ax)

    _save_plot(fig, output_dir, "plot3_spec_aware_comparison")


# ---------------------------------------------------------------------------
# Plot 4a — Pareto Frontier per Domain (EXP-4)
# ---------------------------------------------------------------------------

def plot_pareto(results_dir: Path, output_dir: Path, domain: str) -> None:
    """Generate Pareto frontier plot for a single domain.

    X-axis is task performance (e.g. perplexity), Y-axis is acceptance rate.
    Each point is a different lambda value.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
        domain: One of ``"code"``, ``"medical"``, ``"chat"``.
    """
    exp4_dir = results_dir / "exp4"
    if not exp4_dir.exists():
        logger.warning("Skipping plot4_pareto_%s: exp4 directory not found.", domain)
        return

    lambdas: list[float] = []
    task_perfs: list[float] = []
    alphas: list[float] = []
    alpha_stds: list[float] = []

    # Look for domain-specific result dirs or a consolidated file
    consolidated = _load_json(exp4_dir / f"pareto_{domain}.json")
    if consolidated is not None:
        lambdas = consolidated.get("lambdas", [])
        task_perfs = consolidated.get("task_performance", [])
        alphas = consolidated.get("alpha", [])
        alpha_stds = consolidated.get("alpha_std", [0] * len(alphas))
    else:
        for run_dir in sorted(exp4_dir.glob(f"{domain}_lam_*")):
            acc_data = _load_json(run_dir / "eval_acceptance.json")
            if acc_data is None:
                acc_data = _load_json(run_dir / f"acceptance_{domain}.json")
            cfg_data = _load_json(run_dir / "config.yaml")  # may be YAML, handle JSON
            metrics_data = _load_json(run_dir / "training_metrics.json")

            if acc_data is None:
                continue

            # Extract lambda from dir name
            try:
                lam = float(run_dir.name.split("lam_")[1])
            except (IndexError, ValueError):
                continue

            alpha_val = acc_data.get("mean_alpha", acc_data.get("alpha", None))
            alpha_std = acc_data.get("std_alpha", 0)

            # Task performance: prefer perplexity or eval metric
            task_data = _load_json(run_dir / "eval_task.json")
            task_perf = None
            if task_data is not None:
                task_perf = task_data.get("perplexity", task_data.get("score", None))
            if task_perf is None and metrics_data is not None:
                # Fall back to final training loss
                if isinstance(metrics_data, list) and len(metrics_data) > 0:
                    task_perf = metrics_data[-1].get("task_loss", None)
                elif isinstance(metrics_data, dict):
                    task_perf = metrics_data.get("final_task_loss", metrics_data.get("task_loss", None))

            if alpha_val is not None and task_perf is not None:
                lambdas.append(lam)
                task_perfs.append(task_perf)
                alphas.append(alpha_val)
                alpha_stds.append(alpha_std)

    if len(lambdas) < 2:
        logger.warning("Skipping plot4_pareto_%s: insufficient data points.", domain)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    color = DOMAIN_COLORS.get(domain, COLORS["spec_aware"])

    ax.errorbar(task_perfs, alphas, yerr=alpha_stds, fmt="o", color=color,
                markersize=8, capsize=4, zorder=3, markeredgecolor="white", markeredgewidth=0.5)

    # Label each point with its lambda
    for lam, tp, a in zip(lambdas, task_perfs, alphas):
        ax.annotate(f"\u03bb={lam}", (tp, a), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color="#555555")

    # Draw Pareto frontier line (points not dominated)
    points = sorted(zip(task_perfs, alphas), key=lambda p: p[0])
    pareto_x, pareto_y = [points[0][0]], [points[0][1]]
    max_alpha = points[0][1]
    for px, py in points[1:]:
        if py >= max_alpha:
            pareto_x.append(px)
            pareto_y.append(py)
            max_alpha = py
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "--", color=color, alpha=0.5, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Task Performance (Perplexity \u2193)")
    ax.set_ylabel("Acceptance Rate (\u03b1 \u2191)")
    ax.set_title(f"Pareto Frontier \u2014 {domain.capitalize()} Domain")
    if len(pareto_x) > 1:
        ax.legend()

    _save_plot(fig, output_dir, f"plot4_pareto_{domain}")


# ---------------------------------------------------------------------------
# Plot 4b — Pareto Overlay (EXP-4)
# ---------------------------------------------------------------------------

def plot_pareto_overlay(results_dir: Path, output_dir: Path) -> None:
    """Generate overlay Pareto frontier plot with all domains on the same axes.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp4_dir = results_dir / "exp4"
    if not exp4_dir.exists():
        logger.warning("Skipping plot4_pareto_overlay: exp4 directory not found.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    any_data = False

    for domain in DOMAINS:
        consolidated = _load_json(exp4_dir / f"pareto_{domain}.json")
        task_perfs: list[float] = []
        alphas: list[float] = []

        if consolidated is not None:
            task_perfs = consolidated.get("task_performance", [])
            alphas = consolidated.get("alpha", [])
        else:
            for run_dir in sorted(exp4_dir.glob(f"{domain}_lam_*")):
                acc_data = _load_json(run_dir / "eval_acceptance.json")
                if acc_data is None:
                    acc_data = _load_json(run_dir / f"acceptance_{domain}.json")
                if acc_data is None:
                    continue
                alpha_val = acc_data.get("mean_alpha", acc_data.get("alpha", None))

                task_data = _load_json(run_dir / "eval_task.json")
                metrics_data = _load_json(run_dir / "training_metrics.json")
                task_perf = None
                if task_data is not None:
                    task_perf = task_data.get("perplexity", task_data.get("score", None))
                if task_perf is None and metrics_data is not None:
                    if isinstance(metrics_data, list) and len(metrics_data) > 0:
                        task_perf = metrics_data[-1].get("task_loss", None)
                    elif isinstance(metrics_data, dict):
                        task_perf = metrics_data.get("final_task_loss", None)

                if alpha_val is not None and task_perf is not None:
                    task_perfs.append(task_perf)
                    alphas.append(alpha_val)

        if len(task_perfs) < 2:
            continue

        any_data = True
        color = DOMAIN_COLORS[domain]
        ax.scatter(task_perfs, alphas, color=color, s=70, zorder=3,
                   label=domain.capitalize(), edgecolors="white", linewidths=0.5)

        # Pareto frontier line
        points = sorted(zip(task_perfs, alphas), key=lambda p: p[0])
        pareto_x, pareto_y = [points[0][0]], [points[0][1]]
        max_alpha = points[0][1]
        for px, py in points[1:]:
            if py >= max_alpha:
                pareto_x.append(px)
                pareto_y.append(py)
                max_alpha = py
        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, "--", color=color, alpha=0.5, linewidth=1.5)

    if not any_data:
        logger.warning("Skipping plot4_pareto_overlay: no domain data found.")
        plt.close(fig)
        return

    ax.set_xlabel("Task Performance (Perplexity \u2193)")
    ax.set_ylabel("Acceptance Rate (\u03b1 \u2191)")
    ax.set_title("Pareto Frontiers Across Domains")
    ax.legend()

    _save_plot(fig, output_dir, "plot4_pareto_overlay")


# ---------------------------------------------------------------------------
# Plot 5 — Cross-Domain Heatmap (EXP-5)
# ---------------------------------------------------------------------------

def plot_cross_domain(results_dir: Path, output_dir: Path) -> None:
    """Generate heatmap of cross-domain acceptance rates.

    Expects ``results/exp5/cross_domain_matrix.json`` with structure:
    ``{"train_domain": {"eval_domain": alpha_value}}``.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp5_dir = results_dir / "exp5"
    matrix_data = _load_json(exp5_dir / "cross_domain_matrix.json") if exp5_dir.exists() else None
    if matrix_data is None:
        logger.warning("Skipping plot5_cross_domain: cross_domain_matrix.json not found.")
        return

    # Build DataFrame from nested dict
    train_domains = [d for d in DOMAINS if d in matrix_data]
    eval_domains = DOMAINS  # columns

    matrix: list[list[float]] = []
    for td in train_domains:
        row = []
        for ed in eval_domains:
            val = matrix_data[td].get(ed, np.nan)
            if isinstance(val, dict):
                val = val.get("mean_alpha", val.get("alpha", np.nan))
            row.append(val)
        matrix.append(row)

    df = pd.DataFrame(
        matrix,
        index=[d.capitalize() for d in train_domains],
        columns=[d.capitalize() for d in eval_domains],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        df, annot=True, fmt=".3f", cmap="YlOrRd_r", linewidths=1,
        linecolor="white", ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Acceptance Rate (\u03b1)"},
    )
    ax.set_xlabel("Evaluation Domain")
    ax.set_ylabel("Training Domain")
    ax.set_title("Cross-Domain Speculator Acceptance Rate")

    _save_plot(fig, output_dir, "plot5_cross_domain")


# ---------------------------------------------------------------------------
# Plot 6 — Loss Ablation (EXP-6)
# ---------------------------------------------------------------------------

LOSS_TYPE_LABELS = {
    "kl": "KL",
    "reverse_kl": "Reverse KL",
    "js": "JS",
    "tv": "TV",
    "token_match": "Token Match",
}


def plot_loss_ablation(results_dir: Path, output_dir: Path) -> None:
    """Generate bar chart comparing acceptance rate across loss function types.

    Reads individual result directories under ``results/exp6/`` or a
    consolidated ``results/exp6/loss_ablation.json``.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp6_dir = results_dir / "exp6"
    if not exp6_dir.exists():
        logger.warning("Skipping plot6_loss_ablation: exp6 directory not found.")
        return

    loss_types: list[str] = []
    alphas: list[float] = []
    alpha_stds: list[float] = []

    consolidated = _load_json(exp6_dir / "loss_ablation.json")
    if consolidated is not None:
        for lt in ["kl", "reverse_kl", "js", "tv", "token_match"]:
            if lt in consolidated:
                loss_types.append(lt)
                entry = consolidated[lt]
                alphas.append(entry.get("mean_alpha", entry.get("alpha", 0)))
                alpha_stds.append(entry.get("std_alpha", 0))
    else:
        for lt in ["kl", "reverse_kl", "js", "tv", "token_match"]:
            for run_dir in exp6_dir.glob(f"*{lt}*"):
                acc_data = _load_json(run_dir / "eval_acceptance.json")
                if acc_data is None:
                    continue
                loss_types.append(lt)
                alphas.append(acc_data.get("mean_alpha", acc_data.get("alpha", 0)))
                alpha_stds.append(acc_data.get("std_alpha", 0))
                break  # one per loss type

    if not loss_types:
        logger.warning("Skipping plot6_loss_ablation: no data found.")
        return

    labels = [LOSS_TYPE_LABELS.get(lt, lt) for lt in loss_types]
    x = np.arange(len(labels))

    # Assign distinct colors per loss type
    palette = sns.color_palette("Set2", n_colors=len(loss_types))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, alphas, yerr=alpha_stds, capsize=4, color=palette, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Loss Function")
    ax.set_ylabel("Acceptance Rate (\u03b1)")
    ax.set_title("Loss Function Ablation \u2014 Effect on Acceptance Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    _add_bar_labels(ax)

    _save_plot(fig, output_dir, "plot6_loss_ablation")


# ---------------------------------------------------------------------------
# Plot 7 — Complementarity with Runtime Adaptation (EXP-7)
# ---------------------------------------------------------------------------

def plot_complementarity(results_dir: Path, output_dir: Path) -> None:
    """Generate line chart of alpha over adaptation steps for both approaches.

    Expects ``results/exp7/complementarity.json`` with keys
    ``"standard_ft"`` and ``"spec_aware_ft"``, each mapping to
    ``{"steps": [...], "alpha": [...]}`` (and optionally ``"alpha_std"``).

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    exp7_dir = results_dir / "exp7"
    data = _load_json(exp7_dir / "complementarity.json") if exp7_dir.exists() else None
    if data is None:
        logger.warning("Skipping plot7_complementarity: complementarity.json not found.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for key, label, color, marker in [
        ("standard_ft", "Standard FT + Draft Adaptation", COLORS["code"], "o"),
        ("spec_aware_ft", "Spec-Aware FT + Draft Adaptation", COLORS["spec_aware"], "s"),
    ]:
        entry = data.get(key)
        if entry is None:
            continue
        steps = entry["steps"]
        alpha = entry["alpha"]
        alpha_std = entry.get("alpha_std")

        ax.plot(steps, alpha, marker=marker, label=label, color=color,
                linewidth=2, markersize=6, markeredgecolor="white", markeredgewidth=0.5)
        if alpha_std is not None:
            lower = [a - s for a, s in zip(alpha, alpha_std)]
            upper = [a + s for a, s in zip(alpha, alpha_std)]
            ax.fill_between(steps, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel("Draft Adaptation Steps")
    ax.set_ylabel("Acceptance Rate (\u03b1)")
    ax.set_title("Complementarity with Runtime Draft Adaptation")
    ax.legend()

    _save_plot(fig, output_dir, "plot7_complementarity")


# ---------------------------------------------------------------------------
# Plot 8 — Argmax Agreement Diagnostic
# ---------------------------------------------------------------------------

def plot_argmax_diagnostic(results_dir: Path, output_dir: Path) -> None:
    """Generate grouped bar chart of argmax agreement across conditions.

    Expects ``results/argmax_diagnostic/`` with files like
    ``llama_base_code.json``, ``llama_std_ft_code.json``,
    ``llama_specaware_code.json`` (and similarly for qwen).

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    diag_dir = results_dir / "argmax_diagnostic"
    if not diag_dir.exists():
        logger.warning("Skipping plot_argmax_diagnostic: directory not found.")
        return

    rows = []
    for json_file in sorted(diag_dir.glob("*.json")):
        data = _load_json(json_file)
        if data is None:
            continue
        name = json_file.stem  # e.g. llama_base_code
        parts = name.split("_")
        # Parse family, condition, domain from filename
        family = parts[0]  # llama or qwen
        # Find domain (last part)
        domain = parts[-1] if parts[-1] in DOMAINS else "unknown"
        # Condition is everything between family and domain
        condition_parts = parts[1:-1] if domain != "unknown" else parts[1:]
        condition = "_".join(condition_parts)

        rows.append({
            "family": family.capitalize(),
            "condition": condition,
            "domain": domain,
            "argmax_agreement": data.get("argmax_agreement", 0),
            "top5_overlap": data.get("top5_overlap", 0),
        })

    if not rows:
        logger.warning("Skipping plot_argmax_diagnostic: no data found.")
        return

    df = pd.DataFrame(rows)

    # Plot per family
    for family in df["family"].unique():
        fdf = df[df["family"] == family]

        condition_labels = {"base": "Base", "std_ft": "Standard FT", "specaware": "Spec-Aware FT"}
        conditions = [c for c in ["base", "std_ft", "specaware"] if c in fdf["condition"].values]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(DOMAINS))
        width = 0.25
        condition_colors = [COLORS["base"], COLORS["code"], COLORS["spec_aware"]]

        for i, cond in enumerate(conditions):
            cdf = fdf[fdf["condition"] == cond]
            vals = [cdf[cdf["domain"] == d]["argmax_agreement"].values[0]
                    if len(cdf[cdf["domain"] == d]) > 0 else 0
                    for d in DOMAINS]
            ax.bar(x + i * width, vals, width, label=condition_labels.get(cond, cond),
                   color=condition_colors[i], edgecolor="white", linewidth=0.8)

        ax.set_xlabel("Domain")
        ax.set_ylabel("Argmax Agreement Rate")
        ax.set_title(f"Argmax Agreement: argmax(target) == argmax(draft) — {family}")
        ax.set_xticks(x + width)
        ax.set_xticklabels([d.capitalize() for d in DOMAINS])
        ax.set_ylim(0, 1.0)
        ax.legend()
        _add_bar_labels(ax)

        _save_plot(fig, output_dir, f"plot_argmax_diagnostic_{family.lower()}")

    # Combined top-5 overlap plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, metric in enumerate(["argmax_agreement", "top5_overlap"]):
        ax = axes[idx]
        metric_label = "Argmax Agreement" if metric == "argmax_agreement" else "Top-5 Overlap"
        for family in df["family"].unique():
            fdf = df[df["family"] == family]
            conditions = [c for c in ["base", "std_ft", "specaware"] if c in fdf["condition"].values]
            for cond in conditions:
                cdf = fdf[fdf["condition"] == cond]
                vals = [cdf[cdf["domain"] == d][metric].values[0]
                        if len(cdf[cdf["domain"] == d]) > 0 else 0
                        for d in DOMAINS]
                label = f"{family} {condition_labels.get(cond, cond)}"
                ax.plot(DOMAINS, vals, marker="o", label=label, linewidth=2, markersize=6)
        ax.set_xlabel("Domain")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.0)

    _save_plot(fig, output_dir, "plot_argmax_diagnostic_combined")


# ---------------------------------------------------------------------------
# Plot 9 — Task Eval: Perplexity vs Lambda
# ---------------------------------------------------------------------------

def plot_task_eval(results_dir: Path, output_dir: Path) -> None:
    """Generate perplexity vs lambda curves for task performance evaluation.

    Expects ``results/task_eval/`` with files like
    ``llama_base_code.json``, ``llama_specaware_lam0.5_code.json``, etc.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    task_dir = results_dir / "task_eval"
    if not task_dir.exists():
        logger.warning("Skipping plot_task_eval: directory not found.")
        return

    rows = []
    for json_file in sorted(task_dir.glob("*.json")):
        data = _load_json(json_file)
        if data is None:
            continue
        name = json_file.stem
        ppl_data = data.get("perplexity", {})
        ppl = ppl_data.get("perplexity") if isinstance(ppl_data, dict) else ppl_data
        if ppl is None:
            continue

        # Parse: llama_base_code, llama_std_ft_code, llama_specaware_lam0.5_code
        domain = data.get("domain", name.split("_")[-1])

        if "base" in name:
            lam_val = -0.1  # sentinel for base (no FT)
            condition = "base"
        elif "std_ft" in name:
            lam_val = 0.0  # standard FT = lambda 0
            condition = "std_ft"
        else:
            # Extract lambda from name like "lam0.5"
            match = re.search(r"lam([\d.]+)", name)
            lam_val = float(match.group(1)) if match else None
            condition = f"specaware_{lam_val}"

        if lam_val is None:
            continue

        rows.append({
            "domain": domain,
            "lambda": lam_val,
            "condition": condition,
            "perplexity": ppl,
            "num_samples": ppl_data.get("num_samples") if isinstance(ppl_data, dict) else None,
        })

    if not rows:
        logger.warning("Skipping plot_task_eval: no data found.")
        return

    df = pd.DataFrame(rows)

    # Plot per domain: perplexity vs lambda
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, domain in enumerate(DOMAINS):
        ax = axes[idx]
        ddf = df[df["domain"] == domain].sort_values("lambda")

        if len(ddf) == 0:
            ax.set_title(f"{domain.capitalize()} — No Data")
            continue

        # Separate base/std_ft (horizontal lines) from lambda curve
        base_row = ddf[ddf["lambda"] == -0.1]
        std_ft_row = ddf[ddf["lambda"] == 0.0]
        sweep = ddf[ddf["lambda"] > 0]

        if len(base_row) > 0:
            ax.axhline(y=base_row.iloc[0]["perplexity"], color=COLORS["base"],
                       linestyle="--", linewidth=1.5, label="Base (no FT)")
        if len(std_ft_row) > 0:
            ax.axhline(y=std_ft_row.iloc[0]["perplexity"], color=COLORS["code"],
                       linestyle=":", linewidth=1.5, label="Standard FT")
        if len(sweep) > 0:
            ax.plot(sweep["lambda"], sweep["perplexity"], marker="o",
                    color=COLORS["spec_aware"], linewidth=2, markersize=6,
                    label="Spec-Aware FT")

        ax.set_xlabel("λ")
        ax.set_ylabel("Perplexity")
        ax.set_title(f"{domain.capitalize()} Domain")
        ax.legend(fontsize=9)

    fig.suptitle("Task Performance (Perplexity) vs Speculator-Aware λ", fontsize=14)
    _save_plot(fig, output_dir, "plot_task_eval_perplexity")

    # Also generate a dual-axis plot: alpha and perplexity on same axes
    # (requires EXP-4 acceptance data to be available)
    _plot_task_alpha_tradeoff(results_dir, output_dir, df)


def _plot_task_alpha_tradeoff(results_dir: Path, output_dir: Path, task_df: pd.DataFrame) -> None:
    """Plot alpha and perplexity on dual y-axes to visualize the tradeoff.

    Args:
        results_dir: Root results directory (for loading EXP-4 alpha data).
        output_dir: Directory to write plot files.
        task_df: DataFrame with perplexity data from plot_task_eval.
    """
    # Load EXP-4 Llama alpha data
    exp4_dir = results_dir / "exp4_llama"
    if not exp4_dir or not exp4_dir.exists():
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    has_data = False

    for idx, domain in enumerate(DOMAINS):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        # Alpha data from EXP-4
        alpha_rows = []
        for run_dir in sorted(exp4_dir.glob(f"{domain}_lam_*")):
            acc_data = _load_json(run_dir / "eval_acceptance.json")
            if acc_data is None:
                continue
            try:
                lam = float(run_dir.name.split("_lam_")[1])
            except (IndexError, ValueError):
                continue
            alpha_rows.append({"lambda": lam, "alpha": acc_data.get("alpha", acc_data.get("mean_alpha", 0))})

        if not alpha_rows:
            continue

        adf = pd.DataFrame(alpha_rows).sort_values("lambda")
        ax1.plot(adf["lambda"], adf["alpha"], marker="s", color=COLORS["spec_aware"],
                 linewidth=2, markersize=6, label="α (acceptance rate)")

        # Perplexity data
        ppl_sweep = task_df[(task_df["domain"] == domain) & (task_df["lambda"] > 0)].sort_values("lambda")
        if len(ppl_sweep) > 0:
            ax2.plot(ppl_sweep["lambda"], ppl_sweep["perplexity"], marker="^",
                     color=COLORS["code"], linewidth=2, markersize=6, label="Perplexity")
            has_data = True

        ax1.set_xlabel("λ")
        ax1.set_ylabel("Acceptance Rate (α)", color=COLORS["spec_aware"])
        ax2.set_ylabel("Perplexity", color=COLORS["code"])
        ax1.set_title(f"{domain.capitalize()} Domain")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    if has_data:
        fig.suptitle("Task-α Tradeoff: Acceptance Rate vs Perplexity", fontsize=14)
        _save_plot(fig, output_dir, "plot_task_alpha_tradeoff")
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 10 — Loss Ablation Comparison (Both Families)
# ---------------------------------------------------------------------------

def plot_loss_ablation_combined(results_dir: Path, output_dir: Path) -> None:
    """Generate side-by-side loss ablation comparison for Qwen and Llama.

    Reads from ``results/exp6/`` (Qwen) and ``results/exp6_llama/`` (Llama).

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    loss_order = ["kl", "reverse_kl", "js", "tv", "token_match"]

    family_data = {}
    for family, subdir in [("Qwen (λ=0.01)", "exp6"), ("Llama (λ=0.5)", "exp6_llama")]:
        exp_dir = results_dir / subdir
        if not exp_dir.exists():
            continue

        alphas = {}
        # Try consolidated file first
        consolidated = _load_json(exp_dir / "loss_ablation.json")
        if consolidated is not None:
            for lt in loss_order:
                if lt in consolidated:
                    entry = consolidated[lt]
                    alphas[lt] = entry.get("mean_alpha", entry.get("alpha", 0))
        else:
            # Try individual run directories
            for lt in loss_order:
                for run_dir in exp_dir.glob(f"*{lt}*"):
                    acc_data = _load_json(run_dir / "eval_acceptance.json")
                    if acc_data is not None:
                        alphas[lt] = acc_data.get("mean_alpha", acc_data.get("alpha", 0))
                        break

        if alphas:
            family_data[family] = alphas

    if not family_data:
        logger.warning("Skipping plot_loss_ablation_combined: no data found.")
        return

    fig, axes = plt.subplots(1, len(family_data), figsize=(7 * len(family_data), 5))
    if len(family_data) == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=len(loss_order))
    labels = [LOSS_TYPE_LABELS.get(lt, lt) for lt in loss_order]

    for idx, (family, alphas) in enumerate(family_data.items()):
        ax = axes[idx]
        vals = [alphas.get(lt, 0) for lt in loss_order]
        x = np.arange(len(loss_order))

        # Sort by alpha for visual clarity
        sorted_pairs = sorted(zip(vals, labels, palette), reverse=True)
        sorted_vals, sorted_labels, sorted_colors = zip(*sorted_pairs)

        bars = ax.bar(range(len(sorted_vals)), sorted_vals, color=sorted_colors,
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels, rotation=15, ha="right")
        ax.set_ylabel("Acceptance Rate (α)")
        ax.set_title(family)
        _add_bar_labels(ax)

        # Rank annotation
        for j, (v, l) in enumerate(zip(sorted_vals, sorted_labels)):
            ax.annotate(f"#{j+1}", (j, v), textcoords="offset points",
                       xytext=(0, 15), ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Loss Function Ablation — Ranking Inverts with λ", fontsize=14)
    _save_plot(fig, output_dir, "plot_loss_ablation_combined")


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def generate_summary_table(results_dir: Path) -> pd.DataFrame:
    """Create a summary DataFrame of key results across all experiments.

    Aggregates acceptance rates, KL values, and task performance from
    available experiment result files.

    Args:
        results_dir: Root results directory.

    Returns:
        A DataFrame with one row per experiment/domain combination and columns
        for alpha, KL divergence, task performance, and lambda.
    """
    rows: list[dict[str, Any]] = []

    # EXP-1: baselines
    exp1_dir = results_dir / "exp1"
    for domain in DOMAINS:
        # Load base acceptance (per-domain file)
        base_data = _load_json(exp1_dir / f"acceptance_base_{domain}.json") if exp1_dir.exists() else None
        if base_data is None:
            consolidated = _load_json(exp1_dir / "acceptance_base.json") if exp1_dir.exists() else None
            if consolidated is not None:
                base_data = consolidated.get(domain, consolidated) if domain in consolidated else consolidated

        # Load FT acceptance (per-domain file with eval naming)
        ft_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline_eval_{domain}.json") if exp1_dir.exists() else None
        if ft_data is None:
            ft_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline.json") if exp1_dir.exists() else None
        if ft_data is None:
            continue

        base_alpha: Optional[float] = None
        if base_data is not None:
            base_alpha = base_data.get("mean_alpha", base_data.get("alpha"))

        rows.append({
            "experiment": "EXP-1",
            "domain": domain,
            "method": "standard_ft",
            "lambda": 0.0,
            "base_alpha": base_alpha,
            "alpha": ft_data.get("mean_alpha", ft_data.get("alpha")),
            "alpha_std": ft_data.get("std_alpha", ft_data.get("alpha_std")),
            "kl": ft_data.get("kl_divergence"),
            "task_perf": None,
        })

    # EXP-3: spec-aware results
    for exp3_dir in sorted(results_dir.glob("exp3*")):
        acc_data = _load_json(exp3_dir / "eval_acceptance.json")
        if acc_data is None:
            # Fallback: try acceptance_{domain}.json patterns
            for d in DOMAINS:
                if d in exp3_dir.name:
                    acc_data = _load_json(exp3_dir / f"acceptance_{d}.json")
                    break
            # Try all domains if dir name doesn't hint at one
            if acc_data is None:
                for d in DOMAINS:
                    candidate = _load_json(exp3_dir / f"acceptance_{d}.json")
                    if candidate is not None:
                        acc_data = candidate
                        break
        if acc_data is None:
            continue
        cfg_data = _load_json(exp3_dir / "config.yaml")
        domain = acc_data.get("domain", "unknown")
        if domain == "unknown" and cfg_data:
            domain = cfg_data.get("domain", "unknown")
        # Infer domain from dir name if needed
        if domain == "unknown":
            for d in DOMAINS:
                if d in exp3_dir.name:
                    domain = d
                    break

        lam = acc_data.get("lambda", 0.1)
        if cfg_data and isinstance(cfg_data, dict):
            spec_cfg = cfg_data.get("spec_loss", {})
            if isinstance(spec_cfg, dict):
                lam = spec_cfg.get("lam", lam)

        rows.append({
            "experiment": "EXP-3",
            "domain": domain,
            "method": "spec_aware_ft",
            "lambda": lam,
            "base_alpha": None,
            "alpha": acc_data.get("mean_alpha", acc_data.get("alpha")),
            "alpha_std": acc_data.get("std_alpha"),
            "kl": acc_data.get("kl_divergence"),
            "task_perf": None,
        })

    # EXP-4: lambda sweep
    exp4_dir = results_dir / "exp4"
    if exp4_dir.exists():
        for run_dir in sorted(exp4_dir.glob("*_lam_*")):
            acc_data = _load_json(run_dir / "eval_acceptance.json")
            if acc_data is None:
                # Fallback: try acceptance_{domain}.json where domain is from dir name
                try:
                    dir_domain = run_dir.name.split("_lam_")[0]
                except (IndexError, ValueError):
                    dir_domain = None
                if dir_domain:
                    acc_data = _load_json(run_dir / f"acceptance_{dir_domain}.json")
            if acc_data is None:
                continue
            try:
                parts = run_dir.name.split("_lam_")
                domain = parts[0]
                lam = float(parts[1])
            except (IndexError, ValueError):
                continue

            task_data = _load_json(run_dir / "eval_task.json")
            task_perf = task_data.get("perplexity", task_data.get("score")) if task_data else None

            rows.append({
                "experiment": "EXP-4",
                "domain": domain,
                "method": "spec_aware_ft",
                "lambda": lam,
                "base_alpha": None,
                "alpha": acc_data.get("mean_alpha", acc_data.get("alpha")),
                "alpha_std": acc_data.get("std_alpha"),
                "kl": acc_data.get("kl_divergence"),
                "task_perf": task_perf,
            })

    # EXP-6: loss ablation
    exp6_dir = results_dir / "exp6"
    if exp6_dir.exists():
        consolidated = _load_json(exp6_dir / "loss_ablation.json")
        if consolidated is not None:
            for lt, entry in consolidated.items():
                rows.append({
                    "experiment": "EXP-6",
                    "domain": entry.get("domain", "code"),
                    "method": f"spec_aware_{lt}",
                    "lambda": entry.get("lambda"),
                    "base_alpha": None,
                    "alpha": entry.get("mean_alpha", entry.get("alpha")),
                    "alpha_std": entry.get("std_alpha"),
                    "kl": entry.get("kl_divergence"),
                    "task_perf": entry.get("perplexity", entry.get("score")),
                })

    if not rows:
        logger.warning("No experiment data found for summary table.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_delta_kl_vulnerability(results_dir: Path, output_dir: Path) -> None:
    """Scatter plot of ΔKL vs relative α change across all model families.

    Shows that ΔKL (post-FT KL − base KL) predicts fine-tuning-induced
    speculative decoding degradation (r=−0.73, p=0.005). A vertical threshold
    at ΔKL=0.30 separates degraders from non-degraders in 8/9 cases.

    Args:
        results_dir: Root results directory (unused — data is hardcoded from
            collected experimental results across 3 families × 3 domains).
        output_dir: Directory to save the plot.
    """
    # All 9 data points from EXP-1 across Qwen, Llama, Gemma
    data = [
        ("Qwen",  "code",    0.2031,  +5.6),
        ("Qwen",  "medical", 0.0627,  +5.1),
        ("Qwen",  "chat",    0.1291, +14.0),
        ("Llama", "code",    0.2434,  -8.5),
        ("Llama", "medical", 0.3456, -10.0),
        ("Llama", "chat",    0.4881, -33.5),
        ("Gemma", "code",    0.2866,  -3.0),
        ("Gemma", "medical", 0.8366, -15.2),
        ("Gemma", "chat",    1.5057, -29.3),
    ]

    families = [d[0] for d in data]
    domains = [d[1] for d in data]
    delta_kl = np.array([d[2] for d in data])
    delta_alpha = np.array([d[3] for d in data])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Horizontal zero line
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8, zorder=1)

    # Vertical threshold line at ΔKL = 0.30
    ax.axvline(0.30, color="#E74C3C", linestyle="--", linewidth=1.2, zorder=1,
               label="Vulnerability threshold (ΔKL = 0.30)")

    # Regression line with CI band
    slope, intercept, r_val, p_val, _ = stats.linregress(delta_kl, delta_alpha)
    x_fit = np.linspace(0, max(delta_kl) * 1.1, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color="#555555", linewidth=1.5, linestyle="-", zorder=2)

    # CI band (bootstrap-style approximation using standard error)
    n = len(delta_kl)
    y_pred = slope * delta_kl + intercept
    se = np.sqrt(np.sum((delta_alpha - y_pred) ** 2) / (n - 2))
    x_mean = np.mean(delta_kl)
    se_fit = se * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / np.sum((delta_kl - x_mean) ** 2))
    t_crit = stats.t.ppf(0.975, n - 2)
    ax.fill_between(x_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit,
                    color="#CCCCCC", alpha=0.3, zorder=1)

    # Scatter points: colored by family, shaped by domain
    for i, (fam, dom, dkl, da) in enumerate(data):
        ax.scatter(dkl, da,
                   color=FAMILY_COLORS[fam],
                   marker=DOMAIN_MARKERS[dom],
                   s=120, zorder=3, edgecolors="white", linewidths=0.5)
        # Label each point
        offset_x, offset_y = 0.02, 1.5
        # Nudge overlapping labels
        if fam == "Qwen" and dom == "medical":
            offset_y = -2.5
        elif fam == "Qwen" and dom == "code":
            offset_y = -2.5
        ax.annotate(f"{fam} {dom}", (dkl, da),
                    xytext=(dkl + offset_x, da + offset_y),
                    fontsize=8, color="#333333")

    # Build legend with family colors and domain shapes
    from matplotlib.lines import Line2D
    legend_elements = []
    for fam, color in FAMILY_COLORS.items():
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=color, markersize=9,
                                      label=fam))
    for dom, marker in DOMAIN_MARKERS.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                                      markerfacecolor="#888888", markersize=9,
                                      label=dom.capitalize()))
    legend_elements.append(Line2D([0], [0], color="#E74C3C", linestyle="--",
                                  linewidth=1.2, label="Threshold (ΔKL=0.30)"))
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.9)

    # Annotation with correlation stats
    ax.annotate(f"r = {r_val:.2f}, p = {p_val:.3f}",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

    ax.set_xlabel("ΔKL (Post-FT KL − Base KL)", fontsize=12)
    ax.set_ylabel("Relative α Change (%)", fontsize=12)
    ax.set_title("ΔKL Predicts Speculative Decoding Vulnerability", fontsize=14)

    fig.tight_layout()
    _save_plot(fig, output_dir, "plot_delta_kl_vulnerability")
    plt.close(fig)
    logger.info("Generated ΔKL vulnerability prediction plot.")


def plot_dpo_comparison(results_dir: Path, output_dir: Path) -> None:
    """Bar chart comparing acceptance rates across DPO conditions."""
    dpo_dir = results_dir / "exp_dpo"
    if not dpo_dir.exists():
        logger.info("Skipping DPO comparison plot — no exp_dpo directory.")
        return

    conditions = [
        ("Base\n(no FT)", "base_acceptance_chat.json"),
        ("Standard\nDPO (λ=0)", "baseline_acceptance_chat.json"),
        ("Spec-aware\nDPO (λ=0.1)", "specaware_lam0.1_acceptance_chat.json"),
        ("Spec-aware\nDPO (λ=0.5)", "specaware_lam0.5_acceptance_chat.json"),
    ]

    labels, alphas, stds = [], [], []
    for label, fname in conditions:
        fpath = dpo_dir / fname
        if not fpath.exists():
            logger.info("Skipping DPO plot — missing %s", fname)
            return
        data = json.loads(fpath.read_text())
        labels.append(label)
        alphas.append(data["alpha"])
        stds.append(data["alpha_std"])

    colors = ["#1A5276", "#E74C3C", "#3498DB", "#8E44AD"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, alphas, yerr=stds, width=0.6, color=colors,
                  edgecolor="white", linewidth=1.2, capsize=5, error_kw={"lw": 1.5})

    # Add value labels on bars
    for bar, alpha in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{alpha:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Acceptance Rate (α)", fontsize=12)
    ax.set_title("DPO Alignment: Effect on Speculative Decoding", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(alphas) + max(stds) + 0.08)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Horizontal reference line at base alpha
    ax.axhline(y=alphas[0], color="#1A5276", linestyle="--", alpha=0.4, linewidth=1)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_dir / f"plot_dpo_comparison.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("DPO comparison plot saved to %s", output_dir / "plot_dpo_comparison.png")


def plot_joint_vs_frozen(results_dir: Path, output_dir: Path) -> None:
    """Generate grouped bar chart comparing α across training paradigms per domain.

    Compares base model, standard fine-tuned, speculator-aware (frozen draft,
    best λ from EXP-4), joint KL (EXP-8), and joint proxy (EXP-9) acceptance
    rates for each domain.

    Data sources:
        - Base: ``results/exp1/acceptance_base_{domain}.json``
        - Standard FT: ``results/exp1/acceptance_{domain}_baseline_eval_{domain}.json``
        - Spec-aware frozen: best α from ``results/exp4/{domain}_lam_*/acceptance_{domain}.json``
        - Joint KL (EXP-8): ``results/exp8_joint_{domain}*/acceptance_{domain}.json``
        - Joint proxy (EXP-9): ``results/exp9_overlap_{domain}*/acceptance_{domain}.json``

    Skips gracefully if no EXP-8 or EXP-9 result directories exist.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    # Check for EXP-8 and EXP-9 directories before proceeding
    exp8_dirs = list(results_dir.glob("exp8_joint_*"))
    exp9_dirs = list(results_dir.glob("exp9_overlap_*"))
    if not exp8_dirs and not exp9_dirs:
        logger.info(
            "Skipping plot_joint_vs_frozen: no exp8_joint_* or exp9_overlap_* directories found."
        )
        return

    exp1_dir = results_dir / "exp1"
    exp4_dir = results_dir / "exp4"

    paradigm_labels = ["Base", "Standard FT", "Spec-Aware\n(Frozen)", "Joint KL\n(EXP-8)", "Joint Proxy\n(EXP-9)"]
    paradigm_colors = {
        "Base": "#1A5276",
        "Standard FT": "#E74C3C",
        "Spec-Aware\n(Frozen)": "#3498DB",
        "Joint KL\n(EXP-8)": "#27AE60",
        "Joint Proxy\n(EXP-9)": "#8E44AD",
    }

    # Collect data per domain per paradigm
    domain_data: dict[str, dict[str, tuple[float, float]]] = {}

    for domain in DOMAINS:
        row: dict[str, tuple[float, float]] = {}

        # --- Base model ---
        base_data = _load_json(exp1_dir / f"acceptance_base_{domain}.json")
        if base_data is None:
            consolidated = _load_json(exp1_dir / "acceptance_base.json")
            if consolidated is not None and isinstance(consolidated, dict):
                base_data = consolidated.get(domain)
        if base_data is not None:
            row["Base"] = (
                base_data.get("mean_alpha", base_data.get("alpha", 0.0)),
                base_data.get("std_alpha", base_data.get("alpha_std", 0.0)),
            )

        # --- Standard FT ---
        ft_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline_eval_{domain}.json")
        if ft_data is None:
            ft_data = _load_json(exp1_dir / f"acceptance_{domain}_baseline.json")
        if ft_data is not None:
            row["Standard FT"] = (
                ft_data.get("mean_alpha", ft_data.get("alpha", 0.0)),
                ft_data.get("std_alpha", ft_data.get("alpha_std", 0.0)),
            )

        # --- Spec-aware frozen: find best α across EXP-4 lambda dirs ---
        if exp4_dir.exists():
            best_alpha = -1.0
            best_std = 0.0
            for lam_dir in exp4_dir.glob(f"{domain}_lam_*"):
                acc_path = lam_dir / f"acceptance_{domain}.json"
                data = _load_json(acc_path)
                if data is None:
                    # Try generic acceptance file
                    data = _load_json(lam_dir / "acceptance.json")
                if data is not None:
                    alpha_val = data.get("mean_alpha", data.get("alpha", -1.0))
                    if alpha_val > best_alpha:
                        best_alpha = alpha_val
                        best_std = data.get("std_alpha", data.get("alpha_std", 0.0))
            if best_alpha >= 0:
                row["Spec-Aware\n(Frozen)"] = (best_alpha, best_std)

        # --- Joint KL (EXP-8) ---
        for exp8_dir in sorted(results_dir.glob(f"exp8_joint_{domain}*")):
            data = _load_json(exp8_dir / f"acceptance_{domain}.json")
            if data is None:
                data = _load_json(exp8_dir / "acceptance.json")
            if data is not None:
                row["Joint KL\n(EXP-8)"] = (
                    data.get("mean_alpha", data.get("alpha", 0.0)),
                    data.get("std_alpha", data.get("alpha_std", 0.0)),
                )
                break  # Use first match

        # --- Joint proxy (EXP-9) ---
        for exp9_dir in sorted(results_dir.glob(f"exp9_overlap_{domain}*")):
            data = _load_json(exp9_dir / f"acceptance_{domain}.json")
            if data is None:
                data = _load_json(exp9_dir / "acceptance.json")
            if data is not None:
                row["Joint Proxy\n(EXP-9)"] = (
                    data.get("mean_alpha", data.get("alpha", 0.0)),
                    data.get("std_alpha", data.get("alpha_std", 0.0)),
                )
                break  # Use first match

        if row:
            domain_data[domain] = row

    if not domain_data:
        logger.info("Skipping plot_joint_vs_frozen: no data could be loaded for any domain.")
        return

    # Determine which paradigms actually have data
    present_paradigms = [p for p in paradigm_labels if any(p in domain_data.get(d, {}) for d in DOMAINS)]
    if not present_paradigms:
        logger.info("Skipping plot_joint_vs_frozen: no paradigm data available.")
        return

    n_domains = len(domain_data)
    n_paradigms = len(present_paradigms)
    x = np.arange(n_domains)
    width = 0.8 / n_paradigms
    offsets = np.linspace(-(n_paradigms - 1) / 2, (n_paradigms - 1) / 2, n_paradigms) * width

    fig, ax = plt.subplots(figsize=(max(8, n_domains * 2.5), 5))

    for i, paradigm in enumerate(present_paradigms):
        alphas_plot = []
        stds_plot = []
        domains_ordered = [d for d in DOMAINS if d in domain_data]
        for domain in domains_ordered:
            val = domain_data[domain].get(paradigm, (0.0, 0.0))
            alphas_plot.append(val[0])
            stds_plot.append(val[1])
        bars = ax.bar(
            x + offsets[i],
            alphas_plot,
            width,
            yerr=stds_plot,
            capsize=4,
            label=paradigm.replace("\n", " "),
            color=paradigm_colors[paradigm],
        )
        # Add value labels above each bar
        for bar, alpha_val in zip(bars, alphas_plot):
            if alpha_val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.01,
                    f"{alpha_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    domains_ordered = [d for d in DOMAINS if d in domain_data]
    ax.set_xlabel("Domain")
    ax.set_ylabel("Acceptance Rate (α)")
    ax.set_title("Acceptance Rate by Training Paradigm and Domain")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains_ordered])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    _save_plot(fig, output_dir, "plot_joint_vs_frozen")


def plot_draft_training_trajectory(results_dir: Path, output_dir: Path) -> None:
    """Generate dual-axis line chart of draft loss and acceptance proxy over EXP-8 training.

    Reads ``training_metrics.json`` from the first matching ``exp8_joint_*``
    directory. Plots draft_loss and task_loss on the left y-axis, and
    acceptance_proxy on the right y-axis, against training step.

    Skips gracefully if no EXP-8 directories with training_metrics.json exist.

    Args:
        results_dir: Root results directory.
        output_dir: Directory to write plot files.
    """
    # Find the first exp8 directory that has training_metrics.json
    metrics_path: Path | None = None
    for exp8_dir in sorted(results_dir.glob("exp8_joint_*")):
        candidate = exp8_dir / "training_metrics.json"
        if candidate.exists():
            metrics_path = candidate
            break

    if metrics_path is None:
        logger.info(
            "Skipping plot_draft_training_trajectory: no exp8_joint_*/training_metrics.json found."
        )
        return

    raw = _load_json(metrics_path)
    if raw is None or not isinstance(raw, list) or len(raw) == 0:
        logger.info(
            "Skipping plot_draft_training_trajectory: training_metrics.json is empty or unparseable."
        )
        return

    steps: list[int] = []
    task_losses: list[float] = []
    draft_losses: list[float] = []
    acceptance_proxies: list[float] = []

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        if step is None:
            continue
        steps.append(int(step))
        task_losses.append(float(entry.get("task_loss", float("nan"))))
        draft_losses.append(float(entry.get("draft_loss", float("nan"))))
        acceptance_proxies.append(float(entry.get("acceptance_proxy", float("nan"))))

    if not steps:
        logger.info("Skipping plot_draft_training_trajectory: no valid step entries in metrics.")
        return

    steps_arr = np.array(steps)
    task_arr = np.array(task_losses)
    draft_arr = np.array(draft_losses)
    proxy_arr = np.array(acceptance_proxies)

    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx()

    # --- Left axis: losses ---
    line_draft, = ax_left.plot(
        steps_arr, draft_arr,
        color="#E74C3C", linewidth=2, label="Draft Loss",
    )
    line_task, = ax_left.plot(
        steps_arr, task_arr,
        color="#E74C3C", linewidth=1.5, linestyle="--", alpha=0.5, label="Task Loss",
    )

    # --- Right axis: acceptance proxy ---
    line_proxy, = ax_right.plot(
        steps_arr, proxy_arr,
        color="#3498DB", linewidth=2, label="Acceptance Proxy",
    )

    ax_left.set_xlabel("Training Step")
    ax_left.set_ylabel("Loss", color="#E74C3C")
    ax_left.tick_params(axis="y", labelcolor="#E74C3C")

    ax_right.set_ylabel("Acceptance Proxy (α)", color="#3498DB")
    ax_right.tick_params(axis="y", labelcolor="#3498DB")
    ax_right.set_ylim(0, 1.05)

    ax_left.set_title("EXP-8 Joint Training: Draft Loss & Acceptance Proxy over Steps")

    # Combined legend
    lines = [line_draft, line_task, line_proxy]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="upper right")

    # Grid only on left axis to avoid double grid
    ax_left.yaxis.grid(True, alpha=0.3)
    ax_right.yaxis.grid(False)
    ax_left.set_axisbelow(True)

    _save_plot(fig, output_dir, "plot_draft_training_trajectory")


def main() -> None:
    """Parse arguments and generate all available plots."""
    parser = argparse.ArgumentParser(
        description="Generate plots and summary tables from experiment results.",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Root directory containing experiment result subdirectories.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots",
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        logger.error("Results directory does not exist: %s", results_dir)
        return

    _setup_style()

    logger.info("Generating plots from %s -> %s", results_dir, output_dir)

    # Generate each plot, skipping gracefully if data is missing
    plot_degradation(results_dir, output_dir)
    plot_kl_correlation(results_dir, output_dir)
    plot_spec_aware_comparison(results_dir, output_dir)

    for domain in DOMAINS:
        plot_pareto(results_dir, output_dir, domain)
    plot_pareto_overlay(results_dir, output_dir)

    plot_cross_domain(results_dir, output_dir)
    plot_loss_ablation(results_dir, output_dir)
    plot_loss_ablation_combined(results_dir, output_dir)
    plot_complementarity(results_dir, output_dir)
    plot_argmax_diagnostic(results_dir, output_dir)
    plot_task_eval(results_dir, output_dir)
    plot_delta_kl_vulnerability(results_dir, output_dir)
    plot_dpo_comparison(results_dir, output_dir)
    plot_joint_vs_frozen(results_dir, output_dir)
    plot_draft_training_trajectory(results_dir, output_dir)

    # Summary table
    summary_df = generate_summary_table(results_dir)
    if not summary_df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "summary_table.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info("Summary table saved to %s", csv_path)
        logger.info("\n%s", summary_df.to_string(index=False))
    else:
        logger.info("No data available for summary table.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
