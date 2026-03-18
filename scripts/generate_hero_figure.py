#!/usr/bin/env python3
"""Generate the 2x2 hero summary figure for the speculator-aware fine-tuning project."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------- Global style ----------
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

# Colors
C_BASE = "#1A5276"
C_STD = "#E74C3C"
C_SPEC = "#27AE60"
C_CODE = "#E74C3C"
C_MED = "#27AE60"
C_CHAT = "#F39C12"
C_QWEN = "#3498DB"
C_LLAMA = "#E74C3C"

fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
fig.suptitle("Speculator-Aware Fine-Tuning: Key Results", fontsize=15, fontweight="bold", y=0.97)

# ========== Panel A: The Problem ==========
ax = axes[0, 0]
domains = ["Code", "Medical", "Chat"]
base_a = [0.5954, 0.4163, 0.3784]
std_a = [0.5449, 0.3747, 0.2517]
drops = ["-8.5%", "-10.0%", "-33.5%"]

x = np.arange(len(domains))
w = 0.32
bars1 = ax.bar(x - w/2, base_a, w, color=C_BASE, label="Base model", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w/2, std_a, w, color=C_STD, label="Standard FT", edgecolor="white", linewidth=0.5)

# Annotate drops
for i, (b, s, d) in enumerate(zip(base_a, std_a, drops)):
    ax.annotate(d, xy=(x[i] + w/2, s), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=9,
                fontweight="bold", color=C_STD)

ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.set_ylabel("Acceptance Rate (alpha)")
ax.set_ylim(0, 0.75)
ax.set_title("The Problem: Standard FT Degrades Speculation")
ax.legend(loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.08, 1.05, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold")

# ========== Panel B: The Solution ==========
ax = axes[0, 1]
spec_a = [0.5881, 0.3925, 0.3554]

w = 0.22
bars1 = ax.bar(x - w, base_a, w, color=C_BASE, label="Base model", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x, std_a, w, color=C_STD, label="Standard FT", edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + w, spec_a, w, color=C_SPEC, label=r"Spec-Aware ($\lambda$=0.5)", edgecolor="white", linewidth=0.5)

# Annotate recovery percentages
recoveries = []
for b, s, sa in zip(base_a, std_a, spec_a):
    drop = b - s
    recov = sa - s
    pct = recov / drop * 100 if drop > 0 else 0
    recoveries.append(f"+{recov:.3f}")

for i, r in enumerate(recoveries):
    ax.annotate(r, xy=(x[i] + w, spec_a[i]), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=8,
                fontweight="bold", color=C_SPEC)

ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.set_ylabel("Acceptance Rate (alpha)")
ax.set_ylim(0, 0.75)
ax.set_title(r"The Solution: Spec-Aware FT Recovers $\alpha$")
ax.legend(loc="upper right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.08, 1.05, "(b)", transform=ax.transAxes, fontsize=13, fontweight="bold")

# ========== Panel C: Model-Family Dependence ==========
ax = axes[1, 0]

qwen_kl = [0.4248, 0.6174, 0.6323, 0.6280, 0.6270, 0.6279]
qwen_a = [0.5203, 0.5426, 0.5494, 0.5552, 0.5472, 0.5495]
llama_kl = [0.3793, 0.6233, 0.6340, 0.6292, 0.6218, 0.6227]
llama_a = [0.5954, 0.5343, 0.5536, 0.5504, 0.5369, 0.5449]

# Scatter
ax.scatter(qwen_kl, qwen_a, color=C_QWEN, s=50, zorder=5, edgecolors="white", linewidth=0.5)
ax.scatter(llama_kl, llama_a, color=C_LLAMA, s=50, zorder=5, edgecolors="white", linewidth=0.5)

# Regression lines
for kl_vals, a_vals, color, label_prefix in [
    (qwen_kl, qwen_a, C_QWEN, "Qwen"),
    (llama_kl, llama_a, C_LLAMA, "Llama"),
]:
    slope, intercept, r_value, _, _ = stats.linregress(kl_vals, a_vals)
    kl_range = np.linspace(min(kl_vals) - 0.02, max(kl_vals) + 0.02, 50)
    ax.plot(kl_range, slope * kl_range + intercept, color=color, linestyle="--", linewidth=1.5,
            label=f"{label_prefix} r={r_value:+.2f}")

ax.set_xlabel("KL Divergence (target || draft)")
ax.set_ylabel("Acceptance Rate (alpha)")
ax.set_title("Model-Family Dependence of KL-alpha Relationship")
ax.legend(loc="best", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.08, 1.05, "(c)", transform=ax.transAxes, fontsize=13, fontweight="bold")

# ========== Panel D: The Lambda Knob ==========
ax = axes[1, 1]

lambdas = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
code_a = [0.5449, 0.5379, 0.5409, 0.5596, 0.5646, 0.5881, 0.6158]
med_a = [0.3747, 0.3952, 0.3869, 0.3952, 0.3817, 0.3925, 0.4320]
chat_a = [0.2517, 0.2556, 0.2635, 0.2624, 0.2941, 0.3554, 0.4063]

base_code = 0.5954
base_med = 0.4163
base_chat = 0.3784

ax.plot(lambdas, code_a, "-o", color=C_CODE, markersize=5, linewidth=1.8, label="Code")
ax.plot(lambdas, med_a, "-s", color=C_MED, markersize=5, linewidth=1.8, label="Medical")
ax.plot(lambdas, chat_a, "-^", color=C_CHAT, markersize=5, linewidth=1.8, label="Chat")

# Dashed base lines
ax.axhline(base_code, color=C_CODE, linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(base_med, color=C_MED, linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(base_chat, color=C_CHAT, linestyle=":", linewidth=1, alpha=0.6)

# Label base lines
ax.text(1.02, base_code, "base", fontsize=8, color=C_CODE, va="center", transform=ax.get_yaxis_transform())
ax.text(1.02, base_med, "base", fontsize=8, color=C_MED, va="center", transform=ax.get_yaxis_transform())
ax.text(1.02, base_chat, "base", fontsize=8, color=C_CHAT, va="center", transform=ax.get_yaxis_transform())

ax.set_xlabel(r"Regularization Strength ($\lambda$)")
ax.set_ylabel("Acceptance Rate (alpha)")
ax.set_title(r"The $\lambda$ Knob: Tuning Speculation Preservation")
ax.legend(loc="lower right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.08, 1.05, "(d)", transform=ax.transAxes, fontsize=13, fontweight="bold")

# ========== Save ==========
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_png = "/Users/ritik/Desktop/speculator aware finetuning/plots/plot_hero_summary.png"
out_pdf = "/Users/ritik/Desktop/speculator aware finetuning/plots/plot_hero_summary.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()
