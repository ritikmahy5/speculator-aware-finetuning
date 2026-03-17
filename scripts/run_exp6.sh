#!/bin/bash
# EXP-6: Loss Function Ablation
# Tests whether alternative divergence measures outperform KL.
#
# Loss variants: kl, reverse_kl, js, tv, token_match
#
# Steps:
#   1. Determine optimal lambda from EXP-4 (code domain)
#   2. Fine-tune with each loss variant at that lambda
#   3. Measure alpha and task performance for each
#   4. Generate loss ablation bar chart
#
# Depends on: EXP-4 (optimal lambda)
#
# Usage:
#   bash scripts/run_exp6.sh
#   bash scripts/run_exp6.sh --lam 0.1              # override lambda
#   bash scripts/run_exp6.sh --loss_type js          # single loss type
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-6: Loss Function Ablation"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# --- Parse optional CLI overrides ---
OVERRIDE_LAM=""
FILTER_LOSS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --lam)       OVERRIDE_LAM="$2"; shift 2 ;;
        --loss_type) FILTER_LOSS="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

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

EXP_DIR="${EXP6_RESULTS_DIR:-results/exp6}"
mkdir -p "$EXP_DIR"

DOMAIN="code"

# ====================================================================
# Step 1: Determine optimal lambda from EXP-4
# ====================================================================
if [ -n "$OVERRIDE_LAM" ]; then
    OPTIMAL_LAM="$OVERRIDE_LAM"
    echo "Using override lambda: $OPTIMAL_LAM"
else
    echo ">>> Step 1: Finding optimal lambda from EXP-4 (code domain)"
    OPTIMAL_LAM=$(python -c "
import json, os

exp4_dir = 'results/exp4'
lambdas = ['0.01', '0.05', '0.1', '0.2', '0.5', '1.0']
best_alpha = -1
best_lam = '0.1'  # default

for lam in lambdas:
    acc_file = os.path.join(exp4_dir, f'code_lam_{lam}', 'acceptance_code.json')
    if os.path.exists(acc_file):
        acc = json.load(open(acc_file))
        if acc['alpha'] > best_alpha:
            best_alpha = acc['alpha']
            best_lam = lam

print(best_lam)
" 2>/dev/null || echo "0.1")
    echo "Optimal lambda for code domain: $OPTIMAL_LAM"
fi
echo ""

# ====================================================================
# Step 2: Fine-tune with each loss variant
# ====================================================================
if [ -n "$FILTER_LOSS" ]; then
    LOSS_TYPES="$FILTER_LOSS"
else
    LOSS_TYPES="kl reverse_kl js tv token_match"
fi

TOTAL=$(echo "$LOSS_TYPES" | wc -w | tr -d ' ')
RUN_NUM=0

for LOSS_TYPE in $LOSS_TYPES; do
    RUN_NUM=$((RUN_NUM + 1))
    RUN_DIR="$EXP_DIR/${LOSS_TYPE}_lam_${OPTIMAL_LAM}"

    echo ">>> Run $RUN_NUM/$TOTAL: loss_type=$LOSS_TYPE, lam=$OPTIMAL_LAM"

    # Skip if results already exist
    if [ -f "$RUN_DIR/acceptance_${DOMAIN}.json" ]; then
        echo "  -> Skipping (results already exist)"
        echo ""
        continue
    fi

    mkdir -p "$RUN_DIR"

    # Fine-tune
    echo "  -> Fine-tuning with loss_type=$LOSS_TYPE..."
    python -m src.train \
        --config "${EXP6_CONFIG:-configs/exp6_loss_ablation.yaml}" \
        --domain "$DOMAIN" \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE" \
        --spec_loss.lam "$OPTIMAL_LAM" \
        --spec_loss.loss_type "$LOSS_TYPE" \
        --output_dir "$RUN_DIR"

    # Measure acceptance rate
    echo "  -> Measuring acceptance rate..."
    python -m src.measure_acceptance \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$RUN_DIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "$RUN_DIR/acceptance_${DOMAIN}.json" \
        --max_new_tokens 128 \
        --draft_k 5 \
        --num_prompts 50 \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"

    # Measure KL divergence
    echo "  -> Measuring divergence..."
    python -m src.measure_kl \
        --target_model "$TARGET_MODEL" \
        --draft_model "$DRAFT_MODEL" \
        --adapter_path "$RUN_DIR/final" \
        --prompts_file configs/eval_prompts.yaml \
        --domain "$DOMAIN" \
        --output "$RUN_DIR/kl_${DOMAIN}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" \
        --draft_device "$DRAFT_DEVICE"

    echo "  -> Complete."
    echo ""
done

# ====================================================================
# Step 3: Generate loss ablation plot
# ====================================================================
echo ">>> Generating plot6_loss_ablation.png"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots

# Print summary
python -c "
import json, os

exp_dir = '$EXP_DIR'
optimal_lam = '$OPTIMAL_LAM'
loss_types = '$LOSS_TYPES'.split()

print('LOSS FUNCTION ABLATION RESULTS (code domain)')
print('=' * 70)
print(f'Lambda: {optimal_lam}')
print()
print(f'{\"Loss Type\":15s} {\"Alpha\":>10s} {\"Alpha Std\":>10s} {\"KL Mean\":>10s}')
print('-' * 70)

best_alpha = -1
best_loss = ''

for lt in loss_types:
    run_dir = os.path.join(exp_dir, f'{lt}_lam_{optimal_lam}')
    acc_file = os.path.join(run_dir, 'acceptance_code.json')
    kl_file = os.path.join(run_dir, 'kl_code.json')

    if os.path.exists(acc_file):
        acc = json.load(open(acc_file))
        alpha = acc['alpha']
        alpha_std = acc['alpha_std']
        kl_mean = '--'
        if os.path.exists(kl_file):
            kl_data = json.load(open(kl_file))
            kl_val = kl_data.get('kl_mean', kl_data.get('kl', {}).get('mean', None))
            if kl_val is not None:
                kl_mean = f'{kl_val:.4f}'
        print(f'{lt:15s} {alpha:10.4f} {alpha_std:10.4f} {kl_mean:>10s}')
        if alpha > best_alpha:
            best_alpha = alpha
            best_loss = lt
    else:
        print(f'{lt:15s}       (not yet run)')

print('-' * 70)
if best_loss:
    print(f'Best loss type: {best_loss} (alpha={best_alpha:.4f})')
print('=' * 70)
"

echo ""
echo "============================================"
echo "  EXP-6 COMPLETE"
echo "============================================"
echo "Results saved in: $EXP_DIR/"
echo "Ablation plot:    plots/plot6_loss_ablation.png"
