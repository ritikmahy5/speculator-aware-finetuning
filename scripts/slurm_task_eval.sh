#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=task_eval
#SBATCH --output=logs/task_eval_%j.out
#SBATCH --error=logs/task_eval_%j.err

# Task Performance Evaluation
# Measures perplexity on held-out data for key checkpoints to quantify
# the task-α tradeoff across lambda values.
#
# Evaluates: base model, standard FT, spec-aware at λ=0.1, λ=0.5, λ=1.0
# Domains: code, medical, chat
# Models: Llama (primary — shows degradation)

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p results/task_eval

echo "============================================"
echo "  Task Performance Evaluation"
echo "============================================"

DEVICE="cuda:0"
TARGET="meta-llama/Llama-3.1-8B-Instruct"
NUM_SAMPLES=200

echo "Target model: $TARGET"
echo "Samples per domain: $NUM_SAMPLES"
echo ""

# ====================================================================
# For each domain, measure perplexity at key lambda values
# ====================================================================
for DOMAIN in code medical chat; do
    echo ""
    echo "========== Domain: $DOMAIN =========="

    # Base model (no adapter)
    OUT="results/task_eval/llama_base_${DOMAIN}.json"
    if [ ! -f "$OUT" ]; then
        echo "  -> Base model perplexity..."
        python -m src.eval_task \
            --target_model "$TARGET" \
            --tasks perplexity \
            --domain "$DOMAIN" \
            --num_samples "$NUM_SAMPLES" \
            --output "$OUT" \
            --target_device "$DEVICE"
    else
        echo "  -> Base model: already exists"
    fi

    # Standard FT (λ=0, from EXP-1)
    ADAPTER="results/exp1_llama/baseline_${DOMAIN}/final"
    OUT="results/task_eval/llama_std_ft_${DOMAIN}.json"
    if [ -d "$ADAPTER" ] && [ ! -f "$OUT" ]; then
        echo "  -> Standard FT perplexity..."
        python -m src.eval_task \
            --target_model "$TARGET" \
            --adapter_path "$ADAPTER" \
            --tasks perplexity \
            --domain "$DOMAIN" \
            --num_samples "$NUM_SAMPLES" \
            --output "$OUT" \
            --target_device "$DEVICE"
    else
        echo "  -> Standard FT: ${ADAPTER} not found or already done"
    fi

    # Spec-aware at λ=0.1
    ADAPTER="results/exp4_llama/${DOMAIN}_lam_0.1/final"
    OUT="results/task_eval/llama_specaware_lam0.1_${DOMAIN}.json"
    if [ -d "$ADAPTER" ] && [ ! -f "$OUT" ]; then
        echo "  -> Spec-aware λ=0.1 perplexity..."
        python -m src.eval_task \
            --target_model "$TARGET" \
            --adapter_path "$ADAPTER" \
            --tasks perplexity \
            --domain "$DOMAIN" \
            --num_samples "$NUM_SAMPLES" \
            --output "$OUT" \
            --target_device "$DEVICE"
    else
        echo "  -> Spec-aware λ=0.1: not found or already done"
    fi

    # Spec-aware at λ=0.5
    ADAPTER="results/exp4_llama/${DOMAIN}_lam_0.5/final"
    OUT="results/task_eval/llama_specaware_lam0.5_${DOMAIN}.json"
    if [ -d "$ADAPTER" ] && [ ! -f "$OUT" ]; then
        echo "  -> Spec-aware λ=0.5 perplexity..."
        python -m src.eval_task \
            --target_model "$TARGET" \
            --adapter_path "$ADAPTER" \
            --tasks perplexity \
            --domain "$DOMAIN" \
            --num_samples "$NUM_SAMPLES" \
            --output "$OUT" \
            --target_device "$DEVICE"
    else
        echo "  -> Spec-aware λ=0.5: not found or already done"
    fi

    # Spec-aware at λ=1.0
    ADAPTER="results/exp4_llama/${DOMAIN}_lam_1.0/final"
    OUT="results/task_eval/llama_specaware_lam1.0_${DOMAIN}.json"
    if [ -d "$ADAPTER" ] && [ ! -f "$OUT" ]; then
        echo "  -> Spec-aware λ=1.0 perplexity..."
        python -m src.eval_task \
            --target_model "$TARGET" \
            --adapter_path "$ADAPTER" \
            --tasks perplexity \
            --domain "$DOMAIN" \
            --num_samples "$NUM_SAMPLES" \
            --output "$OUT" \
            --target_device "$DEVICE"
    else
        echo "  -> Spec-aware λ=1.0: not found or already done"
    fi
done

# ====================================================================
# Summary
# ====================================================================
echo ""
echo "============================================"
echo "  Task Eval COMPLETE"
echo "============================================"
python -c "
import json, os, glob

results_dir = 'results/task_eval'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))

print(f'{'Checkpoint':45s} {'Perplexity':>12s} {'Num Samples':>12s}')
print('-' * 70)
for f in files:
    name = os.path.basename(f).replace('.json', '')
    data = json.load(open(f))
    ppl_data = data.get('perplexity', {})
    if isinstance(ppl_data, dict):
        ppl = ppl_data.get('perplexity', 'N/A')
        n = ppl_data.get('num_samples', 'N/A')
    else:
        ppl = ppl_data
        n = 'N/A'
    if isinstance(ppl, float):
        print(f'{name:45s} {ppl:12.2f} {n:>12}')
    else:
        print(f'{name:45s} {str(ppl):>12s} {str(n):>12s}')
"
