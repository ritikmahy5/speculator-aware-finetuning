#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=exp_dpo
#SBATCH --output=logs/exp_dpo_%j.out
#SBATCH --error=logs/exp_dpo_%j.err

# Speculator-Aware DPO Experiment
# Single H200 (140GB) — all 3 models fit on one GPU:
#   target (~18GB) + reference 8-bit (~9GB) + draft (~2GB) = ~29GB
#
# Runs 4 experiments sequentially:
#   1. Measure base acceptance rates
#   2. Standard DPO (lambda=0.0) -> measure acceptance
#   3. Spec-aware DPO (lambda=0.1) -> measure acceptance
#   4. Spec-aware DPO (lambda=0.5) -> measure acceptance

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp_dpo

TARGET="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"
DEVICE="cuda:0"

echo "============================================"
echo "  Speculator-Aware DPO Experiment"
echo "  Started: $(date)"
echo "============================================"

# ====================================================================
# Step 1: Measure base acceptance rates on chat prompts
# ====================================================================
echo ""
echo ">>> Step 1: Base acceptance rate measurement"

if [ ! -f "results/exp_dpo/base_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/base_acceptance_chat.json \
        --target_device "$DEVICE" \
        --draft_device "$DEVICE"
    echo "  Base acceptance measured."
else
    echo "  Base acceptance already exists, skipping."
fi

# ====================================================================
# Step 2: Standard DPO (lambda=0.0)
# ====================================================================
echo ""
echo ">>> Step 2: Standard DPO training (lambda=0.0)"

if [ ! -d "results/exp_dpo/baseline/final" ]; then
    python -m src.train_dpo \
        --config configs/exp_dpo_baseline.yaml \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "  Standard DPO training complete."
else
    echo "  Standard DPO already trained, skipping."
fi

echo "  Measuring acceptance after standard DPO..."
if [ ! -f "results/exp_dpo/baseline_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path results/exp_dpo/baseline/final \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/baseline_acceptance_chat.json \
        --target_device "$DEVICE" \
        --draft_device "$DEVICE"
fi

# ====================================================================
# Step 3: Spec-aware DPO (lambda=0.1)
# ====================================================================
echo ""
echo ">>> Step 3: Spec-aware DPO (lambda=0.1)"

if [ ! -d "results/exp_dpo/specaware_lam0.1/final" ]; then
    python -m src.train_dpo \
        --config configs/exp_dpo_specaware.yaml \
        --spec_loss.lam 0.1 \
        --output_dir results/exp_dpo/specaware_lam0.1 \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "  Spec-aware DPO (lambda=0.1) training complete."
else
    echo "  Spec-aware DPO (lambda=0.1) already trained, skipping."
fi

echo "  Measuring acceptance after spec-aware DPO (lambda=0.1)..."
if [ ! -f "results/exp_dpo/specaware_lam0.1_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path results/exp_dpo/specaware_lam0.1/final \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/specaware_lam0.1_acceptance_chat.json \
        --target_device "$DEVICE" \
        --draft_device "$DEVICE"
fi

# ====================================================================
# Step 4: Spec-aware DPO (lambda=0.5)
# ====================================================================
echo ""
echo ">>> Step 4: Spec-aware DPO (lambda=0.5)"

if [ ! -d "results/exp_dpo/specaware_lam0.5/final" ]; then
    python -m src.train_dpo \
        --config configs/exp_dpo_specaware.yaml \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "  Spec-aware DPO (lambda=0.5) training complete."
else
    echo "  Spec-aware DPO (lambda=0.5) already trained, skipping."
fi

echo "  Measuring acceptance after spec-aware DPO (lambda=0.5)..."
if [ ! -f "results/exp_dpo/specaware_lam0.5_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path results/exp_dpo/specaware_lam0.5/final \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/specaware_lam0.5_acceptance_chat.json \
        --target_device "$DEVICE" \
        --draft_device "$DEVICE"
fi

# ====================================================================
# Summary
# ====================================================================
echo ""
echo "============================================"
echo "  DPO Experiment COMPLETE"
echo "  Finished: $(date)"
echo "============================================"

python -c "
import json, os, glob

results = {}
for f in sorted(glob.glob('results/exp_dpo/*acceptance*.json')):
    name = os.path.basename(f).replace('.json', '')
    data = json.load(open(f))
    alpha = data.get('mean_acceptance_rate', data.get('acceptance_rate', 'N/A'))
    results[name] = alpha

print(f\"{'Condition':45s} {'Alpha':>10s}\")
print('-' * 57)
for name, alpha in results.items():
    if isinstance(alpha, float):
        print(f'{name:45s} {alpha:10.4f}')
    else:
        print(f'{name:45s} {str(alpha):>10s}')
"
