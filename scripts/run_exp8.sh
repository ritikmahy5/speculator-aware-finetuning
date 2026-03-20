#!/bin/bash
# EXP-8: Joint Draft-Target LoRA Training
# Trains target on task loss (CE) while draft adapts via online distillation (KL).
# Both models get LoRA adapters with separate optimizers.
#
# Usage:
#   bash scripts/run_exp8.sh [DOMAIN] [DRAFT_RANK] [DRAFT_LR]
#   bash scripts/run_exp8.sh chat 8 5e-4
#   sbatch scripts/run_exp8.sh chat 8 5e-4
#
# SLURM settings (used when submitted via sbatch):
#SBATCH --partition=gpu
#SBATCH --account=cs7150.202630
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp8_joint
#SBATCH --output=logs/exp8_%j.out
#SBATCH --error=logs/exp8_%j.err
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Parse arguments ---
DOMAIN=${1:-chat}
DRAFT_RANK=${2:-8}
DRAFT_LR=${3:-5e-4}

echo "============================================"
echo "  EXP-8: Joint Draft-Target Training"
echo "============================================"
echo "Domain: $DOMAIN | Draft rank: $DRAFT_RANK | Draft LR: $DRAFT_LR"
echo "Project directory: $PROJECT_DIR"
echo ""

# --- HPC environment (skip if not on cluster) ---
if command -v module &>/dev/null; then
    module load cuda/12.1 2>/dev/null || true
fi
if [ -n "$SLURM_JOB_ID" ]; then
    conda activate /scratch/mahyavanshi.r/envs/specaware 2>/dev/null || true
    export HF_HOME=/scratch/mahyavanshi.r/.cache/huggingface
fi

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

OUTPUT_DIR="results/exp8_joint_${DOMAIN}_r${DRAFT_RANK}_lr${DRAFT_LR}"
mkdir -p "$OUTPUT_DIR"

# ====================================================================
# Step 1: Joint training
# ====================================================================
echo ">>> Step 1/3: Joint training (domain=$DOMAIN)"
python -m src.train_joint \
    --config configs/exp8_joint_training.yaml \
    --domain "$DOMAIN" \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE" \
    --draft_lora.rank "$DRAFT_RANK" \
    --draft_lora.learning_rate "$DRAFT_LR" \
    --output_dir "$OUTPUT_DIR"

echo ">>> Joint training complete."
echo ""

# ====================================================================
# Step 2: Measure acceptance rate (with dual adapters)
# ====================================================================
echo ">>> Step 2/3: Measuring acceptance rate"

for EVAL_DOMAIN in code medical chat mixed; do
    echo "  -> Acceptance: $EVAL_DOMAIN"
    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$OUTPUT_DIR/final/target_adapter" \
        --draft_adapter_path "$OUTPUT_DIR/final/draft_adapter" \
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
# Step 3: Measure KL divergence (with dual adapters)
# ====================================================================
echo ">>> Step 3/3: Measuring KL divergence"

for EVAL_DOMAIN in code medical chat mixed; do
    echo "  -> KL divergence: $EVAL_DOMAIN"
    python -m src.measure_kl \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$OUTPUT_DIR/final/target_adapter" \
        --draft_adapter_path "$OUTPUT_DIR/final/draft_adapter" \
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
echo "  EXP-8 COMPLETE"
echo "============================================"
echo "Results saved in: $OUTPUT_DIR/"
