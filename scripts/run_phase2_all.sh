#!/bin/bash
# Phase 2: ALL EXP-8 sweeps + EXP-9 proxy loss in ONE H200 session
# 13 sequential runs, ~25 min each ≈ 5.5 hours total
#
#SBATCH --partition=gpu
#SBATCH --account=cs7150.202630
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=phase2_all
#SBATCH --output=logs/phase2_all_%j.out
#SBATCH --error=logs/phase2_all_%j.err
set -e

# --- HPC environment ---
module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

DEVICE="cuda:0"
DRAFT_DEVICE="cuda:0"

# --- Helper: extract model names from config ---
EXP8_CONFIG="configs/exp8_joint_training.yaml"
EXP9_CONFIG="configs/exp9_acceptance_proxy.yaml"
TARGET_MODEL=$(python -c "import yaml; print(yaml.safe_load(open())[models][target])")
DRAFT_MODEL=$(python -c "import yaml; print(yaml.safe_load(open())[models][draft])")

echo "============================================"
echo "  Phase 2: Complete Sweep (1 H200, 8 hours)"
echo "  Started: $(date)"
echo "  Target: $TARGET_MODEL"
echo "  Draft:  $DRAFT_MODEL"
echo "============================================"

RUN_COUNT=0
TOTAL_RUNS=13

run_exp8() {
    local DOMAIN=$1 RANK=$2 LR=$3
    local OUT="results/exp8_joint_${DOMAIN}_r${RANK}_lr${LR}"
    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo ">>> [$RUN_COUNT/$TOTAL_RUNS] EXP-8: domain=$DOMAIN rank=$RANK lr=$LR  ($(date))"
    mkdir -p "$OUT"

    # Train
    python -m src.train_joint \
        --config "$EXP8_CONFIG" \
        --domain "$DOMAIN" \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE" \
        --draft_lora.rank "$RANK" \
        --draft_lora.learning_rate "$LR" \
        --output_dir "$OUT"

    # Measure acceptance + KL on all eval domains
    for EVAL in code medical chat mixed; do
        python -m src.measure_acceptance \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$OUT/final/target_adapter" \
            --draft_adapter_path "$OUT/final/draft_adapter" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL" \
            --output "$OUT/acceptance_${EVAL}.json" \
            --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
            --target_device "$DEVICE" --draft_device "$DRAFT_DEVICE"

        python -m src.measure_kl \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$OUT/final/target_adapter" \
            --draft_adapter_path "$OUT/final/draft_adapter" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL" \
            --output "$OUT/kl_${EVAL}.json" \
            --divergence_types kl js tv \
            --target_device "$DEVICE" --draft_device "$DRAFT_DEVICE"
    done
    echo ">>> [$RUN_COUNT/$TOTAL_RUNS] DONE: $OUT  ($(date))"
}

run_exp9() {
    local DOMAIN=$1 LOSS=$2 LAM=$3
    local OUT="results/exp9_${LOSS}_${DOMAIN}_lam${LAM}"
    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo ">>> [$RUN_COUNT/$TOTAL_RUNS] EXP-9: domain=$DOMAIN loss=$LOSS lam=$LAM  ($(date))"
    mkdir -p "$OUT"

    # Train (frozen-draft framework)
    python -m src.train \
        --config "$EXP9_CONFIG" \
        --domain "$DOMAIN" \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE" \
        --spec_loss.loss_type "$LOSS" \
        --spec_loss.lam "$LAM" \
        --output_dir "$OUT"

    # Measure acceptance + KL on all eval domains
    for EVAL in code medical chat mixed; do
        python -m src.measure_acceptance \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$OUT/final" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL" \
            --output "$OUT/acceptance_${EVAL}.json" \
            --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
            --target_device "$DEVICE" --draft_device "$DRAFT_DEVICE"

        python -m src.measure_kl \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$OUT/final" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL" \
            --output "$OUT/kl_${EVAL}.json" \
            --divergence_types kl js tv \
            --target_device "$DEVICE" --draft_device "$DRAFT_DEVICE"
    done
    echo ">>> [$RUN_COUNT/$TOTAL_RUNS] DONE: $OUT  ($(date))"
}

# =====================================================================
# EXP-8 SWEEPS (7 runs)
# Already done: chat, r8, lr5e-4
# =====================================================================

# Sweep 1: Vary rank (chat domain)
run_exp8 chat 4 5e-4
run_exp8 chat 16 5e-4
run_exp8 chat 32 5e-4

# Sweep 2: Vary LR (chat domain, rank=8)
run_exp8 chat 8 1e-4
run_exp8 chat 8 1e-3

# Sweep 3: Vary domain (rank=8, lr=5e-4)
run_exp8 code 8 5e-4
run_exp8 medical 8 5e-4

# =====================================================================
# EXP-9: PROXY LOSS (6 runs)
# =====================================================================

# Overlap loss across domains
run_exp9 code overlap 0.5
run_exp9 chat overlap 0.5
run_exp9 medical overlap 0.5

# Soft-accept loss across domains
run_exp9 code soft_accept 0.5
run_exp9 chat soft_accept 0.5
run_exp9 medical soft_accept 0.5

echo ""
echo "============================================"
echo "  Phase 2 COMPLETE — $(date)"
echo "  $RUN_COUNT/$TOTAL_RUNS runs finished"
echo "============================================"
echo "EXP-8 results: results/exp8_joint_*/"
echo "EXP-9 results: results/exp9_*/"
