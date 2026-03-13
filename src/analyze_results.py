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
    plot_complementarity(results_dir, output_dir)

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
