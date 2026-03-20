#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=gemma_exp3
#SBATCH --output=logs/gemma_exp3_%j.out
#SBATCH --error=logs/gemma_exp3_%j.err

# Gemma EXP-3: Speculator-aware fine-tuning (λ=0.1)
# Pass domain via: sbatch --export=DOMAIN=code scripts/run_gemma_exp3.sh

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp3_gemma

DOMAIN=${DOMAIN:-code}
TARGET="google/gemma-2-9b-it"
DRAFT="google/gemma-2-2b-it"
OUTDIR="results/exp3_gemma/${DOMAIN}_lam0.1"

echo ">>> Gemma EXP-3: spec-aware FT on $DOMAIN (λ=0.1) ($(date))"

if [ ! -d "$OUTDIR/final" ]; then
    python -m src.train \
        --config configs/exp3_gemma.yaml \
        --domain "$DOMAIN" \
        --spec_loss.lam 0.1 \
        --output_dir "$OUTDIR" \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "Training done ($(date))"
else
    echo "Already trained, skipping."
fi

echo ">>> Measuring acceptance"
if [ ! -f "results/exp3_gemma/acceptance_${DOMAIN}_lam0.1.json" ]; then
    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path "$OUTDIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "results/exp3_gemma/acceptance_${DOMAIN}_lam0.1.json" \
        --target_device cuda:0 \
        --draft_device cuda:0
fi

echo ">>> Measuring KL"
if [ ! -f "results/exp3_gemma/kl_${DOMAIN}_lam0.1.json" ]; then
    python -m src.measure_kl \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path "$OUTDIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "results/exp3_gemma/kl_${DOMAIN}_lam0.1.json" \
        --target_device cuda:0 \
        --draft_device cuda:0
fi

echo "Done: Gemma EXP-3 $DOMAIN ($(date))"
