#!/bin/bash
# EXP-3: Speculator-Aware Fine-Tuning (Core Experiment)
# Tests whether speculator-aware loss (KL regularization) preserves alpha.
#
# Steps:
#   1. Fine-tune code domain with lam=0.1, loss_type=kl
#   2. Measure alpha and task perplexity
#   3. Generate three-bar comparison plot (base vs standard-FT vs spec-aware-FT)
#
# Depends on: EXP-1 (baseline results for comparison)
#
# Usage:
#   bash scripts/run_exp3.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-3: Speculator-Aware Fine-Tuning"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

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

EXP_DIR="results/exp3_spec_aware_code_lam0.1"
mkdir -p "$EXP_DIR"

# ====================================================================
# Step 1: Fine-tune code domain with spec-aware loss (lam=0.1)
# ====================================================================
echo ">>> Step 1/3: Fine-tuning code domain with lam=0.1, loss_type=kl"
python -m src.train \
    --config configs/exp3_spec_aware.yaml \
    --domain code \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE" \
    --spec_loss.lam 0.1 \
    --spec_loss.loss_type kl \
    --output_dir "$EXP_DIR" \
    --logging.measure_acceptance_every 200

echo ">>> Spec-aware fine-tuning complete."
echo ""

# ====================================================================
# Step 2: Measure alpha and KL for spec-aware model
# ====================================================================
echo ">>> Step 2/3: Measuring acceptance rate and KL"

# Measure acceptance on all domains
for DOMAIN in code medical chat mixed; do
    echo "  -> Acceptance: $DOMAIN"
    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$EXP_DIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "$EXP_DIR/acceptance_${DOMAIN}.json" \
        --max_new_tokens 128 \
        --draft_k 5 \
        --num_prompts 50 \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done

# Measure KL divergence
for DOMAIN in code medical chat mixed; do
    echo "  -> KL divergence: $DOMAIN"
    python -m src.measure_kl \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$EXP_DIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "$EXP_DIR/kl_${DOMAIN}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
done

# Also measure base model alpha if not already done by EXP-1
BASE_ACC="results/exp1/acceptance_base_code.json"
if [ ! -f "$BASE_ACC" ]; then
    echo "  -> Measuring base model acceptance (EXP-1 results not found)"
    mkdir -p results/exp1
    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$BASE_ACC" \
        --max_new_tokens 128 \
        --draft_k 5 \
        --num_prompts 50 \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"
fi

echo ">>> Measurements complete."
echo ""

# ====================================================================
# Step 3: Generate comparison plot
# ====================================================================
echo ">>> Step 3/3: Generating plot3_spec_aware_comparison.png"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots

# Print comparison summary
python -c "
import json, os

exp1_dir = 'results/exp1'
exp3_dir = '$EXP_DIR'

print('THREE-WAY COMPARISON (code domain)')
print('=' * 60)

# Base model
base_file = os.path.join(exp1_dir, 'acceptance_base_code.json')
if os.path.exists(base_file):
    base = json.load(open(base_file))
    print(f'Base model alpha:         {base[\"alpha\"]:.4f} +/- {base[\"alpha_std\"]:.4f}')
else:
    print('Base model alpha:         (run EXP-1 first)')

# Standard FT (from EXP-1)
std_file = os.path.join(exp1_dir, 'acceptance_code_baseline_eval_code.json')
if os.path.exists(std_file):
    std = json.load(open(std_file))
    print(f'Standard FT alpha:        {std[\"alpha\"]:.4f} +/- {std[\"alpha_std\"]:.4f}')
else:
    print('Standard FT alpha:        (run EXP-1 first)')

# Spec-aware FT
spec_file = os.path.join(exp3_dir, 'acceptance_code.json')
if os.path.exists(spec_file):
    spec = json.load(open(spec_file))
    print(f'Spec-aware FT alpha:      {spec[\"alpha\"]:.4f} +/- {spec[\"alpha_std\"]:.4f}')

# Decision point
if os.path.exists(base_file) and os.path.exists(std_file) and os.path.exists(spec_file):
    base_a = json.load(open(base_file))['alpha']
    std_a = json.load(open(std_file))['alpha']
    spec_a = json.load(open(spec_file))['alpha']
    improvement = spec_a - std_a
    print()
    print(f'Improvement over standard FT: {improvement:+.4f} ({improvement/max(std_a, 1e-6)*100:+.1f}%)')
    print()
    if improvement >= 0.10:
        print('DECISION: alpha improved >=10% absolute -> proceed to EXP-4 (lambda sweep)')
    elif improvement >= 0.05:
        print('DECISION: alpha improved 5-10% -> try EXP-6 (loss ablation) first')
    else:
        print('DECISION: alpha did not improve sufficiently -> debug L_spec trajectory')

print('=' * 60)
"

echo ""
echo "============================================"
echo "  EXP-3 COMPLETE"
echo "============================================"
echo "Results saved in: $EXP_DIR/"
echo "Comparison plot:  plots/plot3_spec_aware_comparison.png"
