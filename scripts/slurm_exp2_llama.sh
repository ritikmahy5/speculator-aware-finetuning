#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp2_llama
#SBATCH --output=logs/exp2_llama_%j.out
#SBATCH --error=logs/exp2_llama_%j.err

# EXP-2 Llama: KL–Acceptance Rate Correlation
# Fine-tune Llama on code with checkpoints, measure α and KL at each.
# Expects NEGATIVE correlation (opposite of Qwen's positive r=+0.956).

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# Use Llama models
cp configs/models_llama.yaml configs/models.yaml

EXP_DIR="results/exp2_llama"
mkdir -p "$EXP_DIR"

DEVICE="cuda:0"
TARGET="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"

echo "============================================"
echo "  EXP-2 Llama: KL-Acceptance Correlation"
echo "============================================"

# Step 1: Fine-tune with checkpoints at ~25%, 50%, 75%, 100%
# 10K samples, batch=4, grad_accum=4 -> 625 steps. Save every 156.
echo ">>> Step 1/3: Fine-tuning code domain with checkpoints"
python -m src.train \
    --config configs/exp2_correlation_llama.yaml \
    --domain code \
    --target_device "$DEVICE" \
    --draft_device "$DEVICE" \
    --spec_loss.lam 0.0 \
    --output_dir "$EXP_DIR/code_with_checkpoints" \
    --logging.save_checkpoints_every 156

echo ">>> Fine-tuning complete."

# Step 2: Measure at base + each checkpoint
echo ">>> Step 2/3: Measuring alpha and KL/JS/TV at checkpoints"

# Base model
echo "  -> Base model"
python -m src.measure_acceptance \
    --target_model "$TARGET" \
    --draft_model "$DRAFT" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$EXP_DIR/acceptance_base.json" \
    --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
    --target_device "$DEVICE" --draft_device "$DEVICE"

python -m src.measure_kl \
    --target_model "$TARGET" \
    --draft_model "$DRAFT" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$EXP_DIR/kl_base.json" \
    --divergence_types kl js tv \
    --target_device "$DEVICE" --draft_device "$DEVICE"

# Checkpoints
CKPT_DIR="$EXP_DIR/code_with_checkpoints/checkpoints"
FINAL_DIR="$EXP_DIR/code_with_checkpoints/final"
CHECKPOINTS=""
if [ -d "$CKPT_DIR" ]; then
    CHECKPOINTS=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n || true)
fi
if [ -d "$FINAL_DIR" ]; then
    CHECKPOINTS="$CHECKPOINTS $FINAL_DIR"
fi

for CKPT in $CHECKPOINTS; do
    CKPT_NAME=$(basename "$CKPT")
    echo "  -> Checkpoint: $CKPT_NAME"

    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path "$CKPT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$EXP_DIR/acceptance_${CKPT_NAME}.json" \
        --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
        --target_device "$DEVICE" --draft_device "$DEVICE"

    python -m src.measure_kl \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path "$CKPT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$EXP_DIR/kl_${CKPT_NAME}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" --draft_device "$DEVICE"
done

# Step 3: Compute correlation
echo ">>> Step 3/3: Computing correlation"
python -c "
import json, glob, os
import numpy as np

exp_dir = '$EXP_DIR'
alphas, kl_vals, js_vals, tv_vals, labels = [], [], [], [], []

base_acc = json.load(open(os.path.join(exp_dir, 'acceptance_base.json')))
base_kl = json.load(open(os.path.join(exp_dir, 'kl_base.json')))
alphas.append(base_acc.get('alpha', base_acc.get('mean_alpha', 0)))
kl_vals.append(base_kl.get('kl_mean', base_kl.get('kl', {}).get('mean', 0)))
js_vals.append(base_kl.get('js_mean', base_kl.get('js', {}).get('mean', 0)))
tv_vals.append(base_kl.get('tv_mean', base_kl.get('tv', {}).get('mean', 0)))
labels.append('base')

for acc_file in sorted(glob.glob(os.path.join(exp_dir, 'acceptance_step_*.json'))):
    name = os.path.basename(acc_file).replace('acceptance_', '').replace('.json', '')
    kl_file = os.path.join(exp_dir, f'kl_{name}.json')
    if os.path.exists(kl_file):
        acc = json.load(open(acc_file))
        kl = json.load(open(kl_file))
        alphas.append(acc.get('alpha', acc.get('mean_alpha', 0)))
        kl_vals.append(kl.get('kl_mean', kl.get('kl', {}).get('mean', 0)))
        js_vals.append(kl.get('js_mean', kl.get('js', {}).get('mean', 0)))
        tv_vals.append(kl.get('tv_mean', kl.get('tv', {}).get('mean', 0)))
        labels.append(name)

acc_file = os.path.join(exp_dir, 'acceptance_final.json')
kl_file = os.path.join(exp_dir, 'kl_final.json')
if os.path.exists(acc_file) and os.path.exists(kl_file):
    acc = json.load(open(acc_file))
    kl = json.load(open(kl_file))
    alphas.append(acc.get('alpha', acc.get('mean_alpha', 0)))
    kl_vals.append(kl.get('kl_mean', kl.get('kl', {}).get('mean', 0)))
    js_vals.append(kl.get('js_mean', kl.get('js', {}).get('mean', 0)))
    tv_vals.append(kl.get('tv_mean', kl.get('tv', {}).get('mean', 0)))
    labels.append('final')

if len(alphas) >= 3:
    from scipy.stats import pearsonr, spearmanr
    print('LLAMA EXP-2 CORRELATION RESULTS')
    print('=' * 60)
    for name, vals in [('KL', kl_vals), ('JS', js_vals), ('TV', tv_vals)]:
        r_p, p_p = pearsonr(vals, alphas)
        r_s, p_s = spearmanr(vals, alphas)
        print(f'{name} vs alpha:  Pearson r={r_p:.4f} (p={p_p:.4g})  Spearman rho={r_s:.4f} (p={p_s:.4g})')
    print('=' * 60)
    for lbl, a, k in zip(labels, alphas, kl_vals):
        print(f'  {lbl:15s}  alpha={a:.4f}  KL={k:.4f}')
"

echo ""
echo "============================================"
echo "  EXP-2 Llama COMPLETE"
echo "============================================"
