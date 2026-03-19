#!/usr/bin/env python3
"""Generate publication-quality benchmark and argmax agreement plots.

Saves to plots/ directory as both PNG (300 DPI) and PDF (vector).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

# ---------------------------------------------------------------------------
# Colors (from CLAUDE.md spec)
# ---------------------------------------------------------------------------
C_BASE = "#1A5276"      # dark blue
C_STD_FT = "#E74C3C"    # red
C_SPEC05 = "#3498DB"    # light blue (spec-aware)
C_SPEC10 = "#27AE60"    # green

# ===================================================================
# PLOT 1: Benchmark Comparison (2-row x 3-col grid)
# ===================================================================

def plot_benchmark_comparison() -> None:
    benchmarks = ["HumanEval", "MedQA", "MMLU"]
    checkpoints = ["Base", "Std FT", u"Spec \u03bb=0.5", u"Spec \u03bb=1.0"]
    colors = [C_BASE, C_STD_FT, C_SPEC05, C_SPEC10]

    llama_data = {
        "HumanEval": [0.6159, 0.5122, 0.4512, 0.4451],
        "MedQA":     [0.6222, 0.6339, 0.6386, 0.6222],
        "MMLU":      [0.6831, 0.6553, 0.6427, 0.6315],
    }
    qwen_data = {
        "HumanEval": [0.6524, 0.5183, 0.5427, 0.5244],
        "MedQA":     [0.6206, 0.6622, 0.6253, 0.5774],
        "MMLU":      [0.7175, 0.7127, 0.6998, 0.6864],
    }

    families = [("Llama 3.1 8B", llama_data), ("Qwen 2.5 7B", qwen_data)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    fig.suptitle("Downstream Benchmark Performance Across Model Checkpoints",
                 fontsize=15, fontweight='bold', y=0.98)

    x = np.arange(len(checkpoints))

    for row, (family_name, data) in enumerate(families):
        for col, bench in enumerate(benchmarks):
            ax = axes[row, col]
            vals = data[bench]
            bars = ax.bar(x, vals, width=0.62, color=colors,
                          edgecolor='white', linewidth=0.8)

            # Value labels on bars
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.1%}", ha='center', va='bottom', fontsize=8.5,
                        fontweight='medium')

            ax.set_title(f"{family_name} \u2014 {bench}",
                         fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(checkpoints, rotation=25, ha='right',
                               fontsize=9)
            ax.set_ylabel("Accuracy" if col == 0 else "")
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

            # Y-axis range with headroom for labels
            ymin = min(vals) - 0.05
            ymax = max(vals) + 0.04
            ax.set_ylim(max(0, ymin), min(1.0, ymax))

            # Light horizontal gridlines only
            ax.yaxis.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

    # Legend at bottom
    legend_elements = [Patch(facecolor=c, label=l)
                       for c, l in zip(colors, checkpoints)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    for ext in ['png', 'pdf']:
        fig.savefig(PLOTS_DIR / f"plot_benchmark_comparison.{ext}")
    plt.close(fig)
    print("Saved plot_benchmark_comparison.png and .pdf")


# ===================================================================
# PLOT 2: Argmax Agreement
# ===================================================================

def plot_argmax_agreement() -> None:
    domains = ["Code", "Medical", "Chat"]
    conditions = ["Base", "Standard FT", "Spec-Aware"]
    cond_colors = [C_BASE, C_STD_FT, C_SPEC05]

    llama_argmax = {
        "Code":    [0.7699, 0.7576, 0.7896],
        "Medical": [0.7198, 0.6830, 0.7259],
        "Chat":    [0.6771, 0.6548, 0.7012],
    }
    qwen_argmax = {
        "Code":    [0.7516, 0.7393, 0.7966],
        "Medical": [0.7101, 0.6920, 0.7467],
        "Chat":    [0.6493, 0.6634, 0.7253],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Argmax Agreement Between Target and Draft Models",
                 fontsize=15, fontweight='bold', y=1.01)

    bar_width = 0.22
    x = np.arange(len(domains))

    for ax_idx, (family_name, data) in enumerate(
            [("Llama 3.1", llama_argmax), ("Qwen 2.5", qwen_argmax)]):
        ax = axes[ax_idx]
        for i, (cond, color) in enumerate(zip(conditions, cond_colors)):
            vals = [data[d][i] for d in domains]
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, vals, width=bar_width, color=color,
                          edgecolor='white', linewidth=0.8, label=cond)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.1%}", ha='center', va='bottom', fontsize=8.5,
                        fontweight='medium')

        ax.set_title(family_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(domains, fontsize=11)
        ax.set_ylabel("Argmax Agreement" if ax_idx == 0 else "")
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0.60, 0.84)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    for ext in ['png', 'pdf']:
        fig.savefig(PLOTS_DIR / f"plot_argmax_agreement.{ext}")
    plt.close(fig)
    print("Saved plot_argmax_agreement.png and .pdf")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_benchmark_comparison()
    plot_argmax_agreement()
    print(f"All plots saved to {PLOTS_DIR}")
