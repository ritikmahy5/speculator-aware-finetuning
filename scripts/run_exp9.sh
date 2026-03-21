#!/bin/bash
# EXP-9: Direct Acceptance Rate Proxy Loss
# Uses proxy losses (overlap or soft_accept) instead of KL in the frozen-draft framework.
# This tests whether directly optimizing acceptance rate outperforms KL.
#
# Usage:
#   bash scripts/run_exp9.sh [DOMAIN] [LOSS_TYPE] [LAM]
#   bash scripts/run_exp9.sh code overlap 0.5
#   sbatch scripts/run_exp9.sh code soft_accept 0.5
#
# SLURM settings (used when submitted via sbatch):
#SBATCH --partition=gpu
#SBATCH --account=cs7150.202630
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp9_proxy
#SBATCH --output=logs/exp9_%j.out
#SBATCH --error=logs/exp9_%j.err
set -e

# --- HPC environment ---
module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# --- Parse arguments ---
DOMAIN=${1:-code}
LOSS_TYPE=${2:-overlap}
LAM=${3:-0.5}

echo "============================================"
echo "  EXP-9: Acceptance Rate Proxy Loss"
echo "  Started: $(date)"
echo "============================================"
echo "Domain: $DOMAIN | Loss type: $LOSS_TYPE | Lambda: $LAM"
echo ""

# --- Detect device ---
DEVICE="cuda:0"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -ge 2 ]; then
    DRAFT_DEVICE="cuda:1"
    echo "Multi-GPU mode: target on cuda:0, draft on cuda:1"
else
    DRAFT_DEVICE="cuda:0"
    echo "Single-GPU mode: both models on cuda:0"
fi
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# --- Extract model names from config ---
TARGET_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models']['target'])")
DRAFT_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models']['draft'])")
echo "Target model: $TARGET_MODEL"
echo "Draft model:  $DRAFT_MODEL"
echo ""

OUTPUT_DIR="results/exp9_${LOSS_TYPE}_${DOMAIN}_lam${LAM}"
mkdir -p "$OUTPUT_DIR"

# ====================================================================
# Step 1: Fine-tune with proxy loss (frozen-draft framework)
# ====================================================================
echo ">>> Step 1/3: Fine-tuning with proxy loss ($LOSS_TYPE, lam=$LAM)"
python -m src.train \
    --config configs/exp9_acceptance_proxy.yaml \
    --domain "$DOMAIN" \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE" \
    --spec_loss.loss_type "$LOSS_TYPE" \
    --spec_loss.lam "$LAM" \
    --output_dir "$OUTPUT_DIR"

echo ">>> Proxy loss fine-tuning complete."
echo ""

# ====================================================================
# Step 2: Measure acceptance rate
# ====================================================================
echo ">>> Step 2/3: Measuring acceptance rate"

for EVAL_DOMAIN in code medical chat mixed; do
    echo "  -> Acceptance: $EVAL_DOMAIN"
    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$OUTPUT_DIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$EVAL_DOMAIN" \
        --output "$OUTPUT_DIR/acceptance_${EVAL_DOMAIN}.json" \
        --max_new_tokens 128 \
        --draft_k 5 \
        --num_prompts 50 \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done

echo ">>> Acceptance measurements complete."
echo ""

# ====================================================================
# Step 3: Measure KL divergence
# ====================================================================
echo ">>> Step 3/3: Measuring KL divergence"

for EVAL_DOMAIN in code medical chat mixed; do
    echo "  -> KL divergence: $EVAL_DOMAIN"
    python -m src.measure_kl \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$OUTPUT_DIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$EVAL_DOMAIN" \
        --output "$OUTPUT_DIR/kl_${EVAL_DOMAIN}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done

echo ">>> KL measurements complete."
echo ""
echo "============================================"
echo "  EXP-9 COMPLETE"
echo "============================================"
echo "Results saved in: $OUTPUT_DIR/"
