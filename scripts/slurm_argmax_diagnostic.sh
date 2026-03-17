#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --job-name=argmax_diag
#SBATCH --output=logs/argmax_diag_%j.out
#SBATCH --error=logs/argmax_diag_%j.err

# Argmax Agreement Diagnostic
# Measures argmax(target) == argmax(draft) for:
#   - Both model families (Llama, Qwen)
#   - Three conditions: base, standard FT, spec-aware FT
#   - All domains

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p results/argmax_diagnostic

echo "============================================"
echo "  Argmax Agreement Diagnostic"
echo "============================================"

DEVICE="cuda:0"

# ====================================================================
# Llama models
# ====================================================================
TARGET="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"

echo ""
echo ">>> Llama models: $TARGET / $DRAFT"

for DOMAIN in code medical chat; do
    echo ""
    echo "--- Domain: $DOMAIN ---"

    # Base model (no adapter)
    echo "  [1/3] Base model..."
    python -m src.measure_argmax \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "results/argmax_diagnostic/llama_base_${DOMAIN}.json" \
        --target_device "$DEVICE" \
        --draft_device "$DEVICE"

    # Standard FT (from EXP-1)
    ADAPTER_STD="results/exp1_llama/${DOMAIN}_baseline/final"
    if [ -d "$ADAPTER_STD" ]; then
        echo "  [2/3] Standard FT..."
        python -m src.measure_argmax \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "$ADAPTER_STD" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "results/argmax_diagnostic/llama_std_ft_${DOMAIN}.json" \
            --target_device "$DEVICE" \
            --draft_device "$DEVICE"
    else
        echo "  [2/3] Standard FT adapter not found at $ADAPTER_STD, skipping"
    fi

    # Spec-aware FT (from EXP-4 at λ=0.5 — good balance point)
    ADAPTER_SPEC="results/exp4_llama/${DOMAIN}_lam_0.5/final"
    if [ -d "$ADAPTER_SPEC" ]; then
        echo "  [3/3] Spec-aware FT (λ=0.5)..."
        python -m src.measure_argmax \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "$ADAPTER_SPEC" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "results/argmax_diagnostic/llama_specaware_${DOMAIN}.json" \
            --target_device "$DEVICE" \
            --draft_device "$DEVICE"
    else
        echo "  [3/3] Spec-aware adapter not found at $ADAPTER_SPEC, skipping"
    fi
done

# ====================================================================
# Qwen models
# ====================================================================
TARGET="Qwen/Qwen2.5-7B-Instruct"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct"

echo ""
echo ">>> Qwen models: $TARGET / $DRAFT"

for DOMAIN in code medical chat; do
    echo ""
    echo "--- Domain: $DOMAIN ---"

    # Base model
    echo "  [1/3] Base model..."
    python -m src.measure_argmax \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "results/argmax_diagnostic/qwen_base_${DOMAIN}.json" \
        --target_device "$DEVICE" \
        --draft_device "$DEVICE"

    # Standard FT (from EXP-1)
    ADAPTER_STD="results/exp1/${DOMAIN}_baseline/final"
    if [ -d "$ADAPTER_STD" ]; then
        echo "  [2/3] Standard FT..."
        python -m src.measure_argmax \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "$ADAPTER_STD" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "results/argmax_diagnostic/qwen_std_ft_${DOMAIN}.json" \
            --target_device "$DEVICE" \
            --draft_device "$DEVICE"
    else
        echo "  [2/3] Standard FT adapter not found at $ADAPTER_STD, skipping"
    fi

    # Spec-aware FT (from EXP-4 at λ=1.0)
    ADAPTER_SPEC="results/exp4/${DOMAIN}_lam_1.0/final"
    if [ -d "$ADAPTER_SPEC" ]; then
        echo "  [3/3] Spec-aware FT (λ=1.0)..."
        python -m src.measure_argmax \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "$ADAPTER_SPEC" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "results/argmax_diagnostic/qwen_specaware_${DOMAIN}.json" \
            --target_device "$DEVICE" \
            --draft_device "$DEVICE"
    else
        echo "  [3/3] Spec-aware adapter not found at $ADAPTER_SPEC, skipping"
    fi
done

# Print summary
echo ""
echo "============================================"
echo "  Argmax Diagnostic COMPLETE"
echo "============================================"
python -c "
import json, os, glob

results_dir = 'results/argmax_diagnostic'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))

print(f'{'Model':30s} {'Argmax Agree':>14s} {'Top5 Overlap':>14s}')
print('-' * 60)
for f in files:
    name = os.path.basename(f).replace('.json', '')
    data = json.load(open(f))
    agree = data.get('argmax_agreement', 0)
    top5 = data.get('top5_overlap', 0)
    print(f'{name:30s} {agree:14.4f} {top5:14.4f}')
"
