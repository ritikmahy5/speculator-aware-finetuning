#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=gemma_exp1
#SBATCH --output=logs/gemma_exp1_%j.out
#SBATCH --error=logs/gemma_exp1_%j.err

# Third Model Family: Gemma 2 9B/2B baseline degradation
# Mirrors EXP-1 for Gemma: measure base α, fine-tune per domain, measure post-FT α.
# Size ratio: 4.5x (9B/2B) — between Llama (8x) and Qwen (14x)

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

EXP_DIR="results/exp1_gemma"
mkdir -p "$EXP_DIR"

DEVICE="cuda:0"
TARGET="google/gemma-2-9b-it"
DRAFT="google/gemma-2-2b-it"

echo "============================================"
echo "  Gemma 2: Baseline Degradation (EXP-1)"
echo "============================================"
echo "Target: $TARGET"
echo "Draft:  $DRAFT"
echo "Size ratio: 4.5x (between Llama 8x and Qwen 14x)"
echo ""

# Step 1: Measure base acceptance rate for all domains
echo ">>> Step 1: Measuring base acceptance rate"
for DOMAIN in code medical chat; do
    OUT="$EXP_DIR/acceptance_base_${DOMAIN}.json"
    if [ ! -f "$OUT" ]; then
        echo "  -> Base α ($DOMAIN)"
        python -m src.measure_acceptance \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "$OUT" \
            --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
            --target_device "$DEVICE" --draft_device "$DEVICE"
    else
        echo "  -> Base α ($DOMAIN): already exists"
    fi
done

# Step 2: Fine-tune on each domain (standard FT, λ=0)
echo ""
echo ">>> Step 2: Fine-tuning per domain (standard LoRA, λ=0)"
for DOMAIN in code medical chat; do
    ADAPTER_DIR="$EXP_DIR/baseline_${DOMAIN}"
    if [ ! -d "$ADAPTER_DIR/final" ]; then
        echo "  -> Fine-tuning $DOMAIN domain"
        python -m src.train \
            --config configs/exp1_gemma.yaml \
            --domain "$DOMAIN" \
            --target_device "$DEVICE" \
            --draft_device "$DEVICE" \
            --spec_loss.lam 0.0 \
            --output_dir "$ADAPTER_DIR"
    else
        echo "  -> $DOMAIN: already trained"
    fi
done

# Step 3: Measure post-FT acceptance rate
echo ""
echo ">>> Step 3: Measuring post-FT acceptance rate"
for DOMAIN in code medical chat; do
    ADAPTER="$EXP_DIR/baseline_${DOMAIN}/final"
    OUT="$EXP_DIR/acceptance_ft_${DOMAIN}.json"
    if [ -d "$ADAPTER" ] && [ ! -f "$OUT" ]; then
        echo "  -> Post-FT α ($DOMAIN)"
        python -m src.measure_acceptance \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "$ADAPTER" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "$OUT" \
            --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
            --target_device "$DEVICE" --draft_device "$DEVICE"
    else
        echo "  -> Post-FT α ($DOMAIN): adapter not found or already done"
    fi
done

# Step 4: Measure KL divergence
echo ""
echo ">>> Step 4: Measuring KL divergence"
for DOMAIN in code medical chat; do
    # Base KL
    OUT="$EXP_DIR/kl_base_${DOMAIN}.json"
    if [ ! -f "$OUT" ]; then
        python -m src.measure_kl \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "$OUT" \
            --divergence_types kl js tv \
            --target_device "$DEVICE" --draft_device "$DEVICE"
    fi

    # Post-FT KL
    ADAPTER="$EXP_DIR/baseline_${DOMAIN}/final"
    OUT="$EXP_DIR/kl_ft_${DOMAIN}.json"
    if [ -d "$ADAPTER" ] && [ ! -f "$OUT" ]; then
        python -m src.measure_kl \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "$ADAPTER" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$DOMAIN" \
            --output "$OUT" \
            --divergence_types kl js tv \
            --target_device "$DEVICE" --draft_device "$DEVICE"
    fi
done

# Summary
echo ""
echo "============================================"
echo "  Gemma EXP-1 COMPLETE"
echo "============================================"
python -c "
import json, os
exp_dir = '$EXP_DIR'
print(f'{'Domain':10s} {'Base α':>10s} {'FT α':>10s} {'Δ':>10s} {'Relative':>10s}')
print('-' * 55)
for domain in ['code', 'medical', 'chat']:
    base_f = os.path.join(exp_dir, f'acceptance_base_{domain}.json')
    ft_f = os.path.join(exp_dir, f'acceptance_ft_{domain}.json')
    if os.path.exists(base_f) and os.path.exists(ft_f):
        base = json.load(open(base_f))
        ft = json.load(open(ft_f))
        ba = base.get('alpha', base.get('mean_alpha', 0))
        fa = ft.get('alpha', ft.get('mean_alpha', 0))
        delta = fa - ba
        rel = delta / ba * 100
        print(f'{domain:10s} {ba:10.4f} {fa:10.4f} {delta:+10.4f} {rel:+9.1f}%')
    else:
        print(f'{domain:10s} (missing data)')
"
