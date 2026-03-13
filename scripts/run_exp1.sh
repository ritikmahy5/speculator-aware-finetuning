#!/bin/bash
# EXP-1: Baseline Degradation Measurement
# Proves that standard LoRA fine-tuning degrades speculator acceptance rate.
#
# Steps:
#   1. Measure BASE acceptance rate (no adapter) for all domains + mixed
#   2. Measure BASE KL divergence
#   3. Fine-tune on code domain (lam=0.0), measure acceptance + KL
#   4. Repeat for medical and chat domains
#   5. Generate degradation bar chart
#
# Usage:
#   bash scripts/run_exp1.sh                  # run directly
#   sbatch scripts/slurm_singlegpu.sh bash scripts/run_exp1.sh  # via SLURM
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-1: Baseline Degradation Measurement"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# --- Detect device ---
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
    echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    # Use second GPU for draft if available
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

EXP_DIR="${EXP1_RESULTS_DIR:-results/exp1}"
mkdir -p "$EXP_DIR"

# ====================================================================
# Step 1: Measure BASE acceptance rate (no adapter) for all domains
# ====================================================================
echo ">>> Step 1/7: Measuring BASE acceptance rate (no adapter)"
for DOMAIN in code medical chat mixed; do
    echo "  -> Domain: $DOMAIN"
    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "$EXP_DIR/acceptance_base_${DOMAIN}.json" \
        --max_new_tokens 128 \
        --draft_k 5 \
        --num_prompts 50 \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done
echo ">>> BASE acceptance measurement complete."
echo ""

# ====================================================================
# Step 2: Measure BASE KL divergence (no adapter)
# ====================================================================
echo ">>> Step 2/7: Measuring BASE KL divergence (no adapter)"
for DOMAIN in code medical chat mixed; do
    echo "  -> Domain: $DOMAIN"
    python -m src.measure_kl \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "$EXP_DIR/kl_base_${DOMAIN}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done
echo ">>> BASE KL measurement complete."
echo ""

# ====================================================================
# Steps 3-5: Fine-tune + measure for each domain
# ====================================================================
for DOMAIN in code medical chat; do
    STEP_NUM=$((3 + $(echo "code medical chat" | tr ' ' '\n' | grep -n "^${DOMAIN}$" | cut -d: -f1) - 1))

    echo ">>> Step ${STEP_NUM}/7: Fine-tuning on $DOMAIN domain (lam=0.0)"
    python -m src.train \
        --config configs/exp1_baseline.yaml \
        --domain "$DOMAIN" \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE" \
        --spec_loss.lam 0.0 \
        --output_dir "$EXP_DIR/baseline_${DOMAIN}"

    echo "  -> Measuring acceptance rate for ${DOMAIN}-FT model"
    for EVAL_DOMAIN in code medical chat mixed; do
        python -m src.measure_acceptance \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$EXP_DIR/baseline_${DOMAIN}/final" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL_DOMAIN" \
            --output "$EXP_DIR/acceptance_${DOMAIN}_baseline_eval_${EVAL_DOMAIN}.json" \
            --max_new_tokens 128 \
            --draft_k 5 \
            --num_prompts 50 \
            --target_device "$DEVICE" \
            --draft_device "$DRAFT_DEVICE"
    done

    echo "  -> Measuring KL divergence for ${DOMAIN}-FT model"
    for EVAL_DOMAIN in code medical chat mixed; do
        python -m src.measure_kl \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$EXP_DIR/baseline_${DOMAIN}/final" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL_DOMAIN" \
            --output "$EXP_DIR/kl_${DOMAIN}_baseline_eval_${EVAL_DOMAIN}.json" \
            --divergence_types kl js tv \
            --target_device "$DEVICE" \
            --draft_device "$DRAFT_DEVICE"
    done

    echo ">>> ${DOMAIN} domain fine-tuning and measurement complete."
    echo ""
done

# ====================================================================
# Step 7: Generate degradation plot
# ====================================================================
echo ">>> Step 7/7: Generating plot1_degradation.png"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots
echo ""

# ====================================================================
# Summary
# ====================================================================
echo "============================================"
echo "  EXP-1 COMPLETE"
echo "============================================"
echo ""
echo "Results saved in: $EXP_DIR/"
echo ""
echo "Key output files:"
echo "  Base acceptance:     $EXP_DIR/acceptance_base_*.json"
echo "  Base KL:             $EXP_DIR/kl_base_*.json"
echo "  Code baseline:       $EXP_DIR/baseline_code/"
echo "  Medical baseline:    $EXP_DIR/baseline_medical/"
echo "  Chat baseline:       $EXP_DIR/baseline_chat/"
echo "  Degradation plot:    plots/plot1_degradation.png"
echo ""

python -c "
import json, glob, os

print('ACCEPTANCE RATE SUMMARY')
print('=' * 60)
print(f'{\"Config\":30s} {\"Alpha\":>10s} {\"Std\":>10s}')
print('-' * 60)

# Base
for f in sorted(glob.glob('$EXP_DIR/acceptance_base_*.json')):
    d = json.load(open(f))
    domain = os.path.basename(f).replace('acceptance_base_', '').replace('.json', '')
    print(f'Base ({domain:8s})              {d[\"alpha\"]:10.4f} {d[\"alpha_std\"]:10.4f}')

# Fine-tuned (same-domain eval)
for domain in ['code', 'medical', 'chat']:
    f = f'$EXP_DIR/acceptance_{domain}_baseline_eval_{domain}.json'
    if os.path.exists(f):
        d = json.load(open(f))
        print(f'FT-{domain:8s} (eval {domain:8s})  {d[\"alpha\"]:10.4f} {d[\"alpha_std\"]:10.4f}')

print('=' * 60)
"
