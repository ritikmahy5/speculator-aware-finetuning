"""Generate Gemma EXP-1 plots and three-family degradation comparison.

Produces:
  - plots/plot_three_family_degradation.png/.pdf
  - plots/plot_gemma_base_vs_ft.png/.pdf
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style setup (matches project conventions)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
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
    "figure.constrained_layout.use": True,
})

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / name}.png and .pdf")


# =========================================================================
# Plot 1: Three-family relative degradation
# =========================================================================
def plot_three_family_degradation() -> None:
    domains = ["Code", "Medical", "Chat"]
    families = ["Llama", "Gemma", "Qwen"]
    colors = {"Llama": "#E74C3C", "Gemma": "#9B59B6", "Qwen": "#3498DB"}

    # Relative % change from base
    data = {
        "Llama":  [-8.5, -10.0, -33.5],
        "Gemma":  [-3.0, -15.2, -29.3],
        "Qwen":   [+5.6, +5.1, +14.0],
    }

    x = np.arange(len(domains))
    n_families = len(families)
    bar_width = 0.22
    offsets = np.linspace(-(n_families - 1) / 2 * bar_width,
                          (n_families - 1) / 2 * bar_width,
                          n_families)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, family in enumerate(families):
        vals = data[family]
        bars = ax.bar(x + offsets[i], vals, bar_width,
                      label=family, color=colors[family],
                      edgecolor="white", linewidth=0.6, zorder=3)
        # Value labels
        for bar, v in zip(bars, vals):
            y_pos = bar.get_height()
            va = "bottom" if v >= 0 else "top"
            offset = 0.8 if v >= 0 else -0.8
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
                    f"{v:+.1f}%", ha="center", va=va, fontsize=9,
                    fontweight="bold", color=colors[family])

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Relative Change in Acceptance Rate (%)")
    ax.set_title("Speculative Decoding Degradation After Standard LoRA Fine-Tuning",
                 fontsize=13, fontweight="bold", pad=22)
    ax.text(0.5, 1.01, "Three Model Families", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11, fontstyle="italic",
            color="#555555")
    ax.legend(loc="lower left", framealpha=0.9)

    # Extend y range a bit for label room
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, -40), max(ymax, 22))

    save(fig, "plot_three_family_degradation")
    plt.close(fig)


# =========================================================================
# Plot 2: Gemma base vs FT acceptance rates
# =========================================================================
def plot_gemma_base_vs_ft() -> None:
    domains = ["Code", "Medical", "Chat"]
    base_vals = [0.6247, 0.3976, 0.3984]
    ft_vals = [0.6056, 0.3372, 0.2815]

    x = np.arange(len(domains))
    bar_width = 0.30

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars_base = ax.bar(x - bar_width / 2, base_vals, bar_width,
                       label="Base Model", color="#1A5276",
                       edgecolor="white", linewidth=0.6, zorder=3)
    bars_ft = ax.bar(x + bar_width / 2, ft_vals, bar_width,
                     label="After LoRA Fine-Tuning", color="#E74C3C",
                     edgecolor="white", linewidth=0.6, zorder=3)

    # Annotate bars with values
    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, color="#1A5276", fontweight="bold")

    for bar in bars_ft:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, color="#E74C3C", fontweight="bold")

    # Percentage drop annotations (arrows between bar pairs)
    for i in range(len(domains)):
        pct_drop = (ft_vals[i] - base_vals[i]) / base_vals[i] * 100
        mid_x = x[i]
        mid_y = max(base_vals[i], ft_vals[i]) + 0.045
        ax.annotate(f"{pct_drop:.1f}%",
                    xy=(mid_x, mid_y), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#C0392B",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#FADBD8",
                              ec="#E74C3C", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Acceptance Rate ($\\alpha$)")
    ax.set_title("Gemma 2 9B / 2B: Acceptance Rate Degradation (EXP-1)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(base_vals) + 0.10)

    save(fig, "plot_gemma_base_vs_ft")
    plt.close(fig)


# =========================================================================
if __name__ == "__main__":
    plot_three_family_degradation()
    plot_gemma_base_vs_ft()
    print("Done.")
