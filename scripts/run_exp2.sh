#!/bin/bash
# EXP-2: KL-Acceptance Rate Correlation
# Validates that KL divergence predicts acceptance rate (|r| > 0.85).
#
# Steps:
#   1. Fine-tune code domain with checkpoints at 25%, 50%, 75%, 100%
#   2. Measure alpha AND KL/JS/TV at each checkpoint
#   3. Compute correlation and generate scatter plot
#
# Depends on: EXP-1 (for context, but runs independently)
#
# Usage:
#   bash scripts/run_exp2.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-2: KL-Acceptance Rate Correlation"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# --- Detect device ---
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
    echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    if [ "$GPU_COUNT" -ge 2 ]; then
        DRAFT_DEVICE="cuda:1"
        echo "Multi-GPU mode: target on cuda:0, draft on cuda:1"
    else
        DRAFT_DEVICE="cuda:0"
        echo "Single-GPU mode: both models on cuda:0"
    fi
else
    DEVICE="cpu"
    DRAFT_DEVICE="cpu"
    echo "No GPU detected -- running on CPU (will be slow)"
fi

# --- Extract model names from config ---
TARGET_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models']['target'])")
DRAFT_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models']['draft'])")
echo "Target model: $TARGET_MODEL"
echo "Draft model:  $DRAFT_MODEL"
echo ""

EXP_DIR="results/exp2"
mkdir -p "$EXP_DIR"

# ====================================================================
# Step 1: Fine-tune code domain, saving checkpoints at 25/50/75/100%
# ====================================================================
# We compute total steps and set save_checkpoints_every accordingly.
# With 10K samples, batch_size=4, grad_accum=4: effective_batch=16
# Total steps = 10000 / 16 = 625. Save every ~156 steps for 4 checkpoints.
# We use the config which has save_at_fractions: [0.25, 0.50, 0.75, 1.0]
echo ">>> Step 1/3: Fine-tuning code domain with checkpoints"
python -m src.train \
    --config configs/exp2_correlation.yaml \
    --domain code \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE" \
    --spec_loss.lam 0.0 \
    --output_dir "$EXP_DIR/code_with_checkpoints" \
    --logging.save_checkpoints_every 156

echo ">>> Fine-tuning complete."
echo ""

# ====================================================================
# Step 2: Measure alpha and KL/JS/TV at each checkpoint + base model
# ====================================================================
echo ">>> Step 2/3: Measuring alpha and divergence at each checkpoint"

# First measure base model (no adapter) as reference point
echo "  -> Measuring base model (no adapter)"
python -m src.measure_acceptance \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$EXP_DIR/acceptance_base.json" \
    --max_new_tokens 128 \
    --draft_k 5 \
    --num_prompts 50 \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE"

python -m src.measure_kl \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$EXP_DIR/kl_base.json" \
    --divergence_types kl js tv \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE"

# Measure at each checkpoint
CKPT_DIR="$EXP_DIR/code_with_checkpoints/checkpoints"
FINAL_DIR="$EXP_DIR/code_with_checkpoints/final"

# Find all checkpoint directories and the final adapter
CHECKPOINTS=""
if [ -d "$CKPT_DIR" ]; then
    CHECKPOINTS=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n || true)
fi

# Add final checkpoint
if [ -d "$FINAL_DIR" ]; then
    CHECKPOINTS="$CHECKPOINTS $FINAL_DIR"
fi

for CKPT in $CHECKPOINTS; do
    CKPT_NAME=$(basename "$CKPT")
    echo "  -> Checkpoint: $CKPT_NAME"

    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$CKPT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$EXP_DIR/acceptance_${CKPT_NAME}.json" \
        --max_new_tokens 128 \
        --draft_k 5 \
        --num_prompts 50 \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"

    python -m src.measure_kl \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$CKPT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$EXP_DIR/kl_${CKPT_NAME}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done

echo ">>> Checkpoint measurements complete."
echo ""

# ====================================================================
# Step 3: Generate correlation plot
# ====================================================================
echo ">>> Step 3/3: Generating plot2_kl_correlation.png"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots

# Also compute and print correlation statistics
python -c "
import json, glob, os
import numpy as np

exp_dir = '$EXP_DIR'

alphas = []
kl_vals = []
js_vals = []
tv_vals = []
labels = []

# Base model
base_acc = json.load(open(os.path.join(exp_dir, 'acceptance_base.json')))
base_kl = json.load(open(os.path.join(exp_dir, 'kl_base.json')))
alphas.append(base_acc['alpha'])
kl_vals.append(base_kl.get('kl_mean', base_kl.get('kl', {}).get('mean', 0)))
js_vals.append(base_kl.get('js_mean', base_kl.get('js', {}).get('mean', 0)))
tv_vals.append(base_kl.get('tv_mean', base_kl.get('tv', {}).get('mean', 0)))
labels.append('base')

# Checkpoints
for acc_file in sorted(glob.glob(os.path.join(exp_dir, 'acceptance_step_*.json'))):
    name = os.path.basename(acc_file).replace('acceptance_', '').replace('.json', '')
    kl_file = os.path.join(exp_dir, f'kl_{name}.json')
    if os.path.exists(kl_file):
        acc = json.load(open(acc_file))
        kl = json.load(open(kl_file))
        alphas.append(acc['alpha'])
        kl_vals.append(kl.get('kl_mean', kl.get('kl', {}).get('mean', 0)))
        js_vals.append(kl.get('js_mean', kl.get('js', {}).get('mean', 0)))
        tv_vals.append(kl.get('tv_mean', kl.get('tv', {}).get('mean', 0)))
        labels.append(name)

# Final
acc_file = os.path.join(exp_dir, 'acceptance_final.json')
kl_file = os.path.join(exp_dir, 'kl_final.json')
if os.path.exists(acc_file) and os.path.exists(kl_file):
    acc = json.load(open(acc_file))
    kl = json.load(open(kl_file))
    alphas.append(acc['alpha'])
    kl_vals.append(kl.get('kl_mean', kl.get('kl', {}).get('mean', 0)))
    js_vals.append(kl.get('js_mean', kl.get('js', {}).get('mean', 0)))
    tv_vals.append(kl.get('tv_mean', kl.get('tv', {}).get('mean', 0)))
    labels.append('final')

if len(alphas) >= 3:
    from scipy.stats import pearsonr, spearmanr
    print('CORRELATION RESULTS')
    print('=' * 60)
    for name, vals in [('KL', kl_vals), ('JS', js_vals), ('TV', tv_vals)]:
        r_p, p_p = pearsonr(vals, alphas)
        r_s, p_s = spearmanr(vals, alphas)
        print(f'{name} vs alpha:  Pearson r={r_p:.4f} (p={p_p:.4g})  Spearman rho={r_s:.4f} (p={p_s:.4g})')
    print('=' * 60)
    print(f'Data points: {len(alphas)}')
    for lbl, a, k in zip(labels, alphas, kl_vals):
        print(f'  {lbl:15s}  alpha={a:.4f}  KL={k:.4f}')
else:
    print(f'Only {len(alphas)} data points -- not enough for correlation')
"
echo ""
echo "============================================"
echo "  EXP-2 COMPLETE"
echo "============================================"
echo "Results saved in: $EXP_DIR/"
echo "Correlation plot: plots/plot2_kl_correlation.png"
