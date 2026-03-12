#!/bin/bash
# EXP-5: Cross-Domain Analysis
# Compares optimal lambda across domains and tests cross-domain generalization.
#
# Steps:
#   1. Aggregate EXP-4 results to find optimal lambda per domain
#   2. Cross-evaluate: take domain-X FT model, measure alpha on domain-Y prompts
#   3. Generate cross-domain heatmap
#
# Depends on: EXP-4 (need optimal lambda and trained models per domain)
#
# Usage:
#   bash scripts/run_exp5.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-5: Cross-Domain Analysis"
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

EXP_DIR="results/exp5"
mkdir -p "$EXP_DIR"

DOMAINS="code medical chat"

# ====================================================================
# Step 1: Find optimal lambda per domain from EXP-4
# ====================================================================
echo ">>> Step 1/3: Finding optimal lambda per domain from EXP-4 results"

# This script finds the lambda that maximizes alpha while keeping task loss
# within 5% of the baseline (lam=0.0) for each domain.
OPTIMAL_LAMBDAS=$(python -c "
import json, os, sys

exp4_dir = 'results/exp4'
domains = ['code', 'medical', 'chat']
lambdas = ['0.01', '0.05', '0.1', '0.2', '0.5', '1.0']

result = {}
for domain in domains:
    best_alpha = -1
    best_lam = '0.1'  # default fallback
    for lam in lambdas:
        acc_file = os.path.join(exp4_dir, f'{domain}_lam_{lam}', f'acceptance_{domain}.json')
        if os.path.exists(acc_file):
            acc = json.load(open(acc_file))
            if acc['alpha'] > best_alpha:
                best_alpha = acc['alpha']
                best_lam = lam
    result[domain] = best_lam
    print(f'{domain}={best_lam}', end=' ', file=sys.stderr)

# Output as space-separated domain=lambda pairs
for domain in domains:
    print(f'{domain}={result[domain]}')
" 2>&1)

echo "Optimal lambdas: $OPTIMAL_LAMBDAS"
echo ""

# Parse optimal lambdas into variables
CODE_LAM="0.1"
MEDICAL_LAM="0.1"
CHAT_LAM="0.1"
for PAIR in $OPTIMAL_LAMBDAS; do
    DOMAIN_NAME=$(echo "$PAIR" | cut -d= -f1)
    LAM_VAL=$(echo "$PAIR" | cut -d= -f2)
    case $DOMAIN_NAME in
        code)    CODE_LAM="$LAM_VAL" ;;
        medical) MEDICAL_LAM="$LAM_VAL" ;;
        chat)    CHAT_LAM="$LAM_VAL" ;;
    esac
done

echo "Using: code=$CODE_LAM, medical=$MEDICAL_LAM, chat=$CHAT_LAM"
echo ""

# ====================================================================
# Step 2: Cross-domain evaluation
# ====================================================================
echo ">>> Step 2/3: Cross-domain acceptance rate evaluation"
echo "  Training domain -> Evaluation domain matrix"
echo ""

# For each trained model (at optimal lambda), measure alpha on ALL domains
for TRAIN_DOMAIN in code medical chat; do
    # Get the optimal lambda for this training domain
    case $TRAIN_DOMAIN in
        code)    OPT_LAM="$CODE_LAM" ;;
        medical) OPT_LAM="$MEDICAL_LAM" ;;
        chat)    OPT_LAM="$CHAT_LAM" ;;
    esac

    ADAPTER_PATH="results/exp4/${TRAIN_DOMAIN}_lam_${OPT_LAM}/final"

    # Check if adapter exists
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "  WARNING: Adapter not found at $ADAPTER_PATH -- skipping $TRAIN_DOMAIN"
        echo "           Run EXP-4 for domain=$TRAIN_DOMAIN first."
        continue
    fi

    for EVAL_DOMAIN in code medical chat mixed; do
        echo "  -> Train=$TRAIN_DOMAIN (lam=$OPT_LAM), Eval=$EVAL_DOMAIN"

        python -m src.measure_acceptance \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$ADAPTER_PATH" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL_DOMAIN" \
            --output "$EXP_DIR/acceptance_train_${TRAIN_DOMAIN}_eval_${EVAL_DOMAIN}.json" \
            --max_new_tokens 128 \
            --draft_k 5 \
            --num_prompts 50 \
            --target_device "$DEVICE" \
            --draft_device "$DRAFT_DEVICE"

        python -m src.measure_kl \
            --target_model "$TARGET_MODEL" \
            --draft_model "$DRAFT_MODEL" \
            --adapter_path "$ADAPTER_PATH" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$EVAL_DOMAIN" \
            --output "$EXP_DIR/kl_train_${TRAIN_DOMAIN}_eval_${EVAL_DOMAIN}.json" \
            --divergence_types kl js tv \
            --target_device "$DEVICE" \
            --draft_device "$DRAFT_DEVICE"
    done
    echo ""
done

echo ">>> Cross-domain measurements complete."
echo ""

# ====================================================================
# Step 3: Generate cross-domain heatmap
# ====================================================================
echo ">>> Step 3/3: Generating plot5_cross_domain.png"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots

# Print cross-domain matrix
python -c "
import json, os

exp_dir = '$EXP_DIR'
domains = ['code', 'medical', 'chat']
eval_domains = ['code', 'medical', 'chat', 'mixed']

print('CROSS-DOMAIN ACCEPTANCE RATE MATRIX')
print('=' * 70)
header = f'{\"Train \\\\ Eval\":15s}'
for ed in eval_domains:
    header += f' {ed:>10s}'
print(header)
print('-' * 70)

for td in domains:
    row = f'{td:15s}'
    for ed in eval_domains:
        f = os.path.join(exp_dir, f'acceptance_train_{td}_eval_{ed}.json')
        if os.path.exists(f):
            d = json.load(open(f))
            row += f' {d[\"alpha\"]:10.4f}'
        else:
            row += f' {\"--\":>10s}'
    print(row)

print('=' * 70)
print()
print('OPTIMAL LAMBDA PER DOMAIN')
print(f'  code:    $CODE_LAM')
print(f'  medical: $MEDICAL_LAM')
print(f'  chat:    $CHAT_LAM')
"

echo ""
echo "============================================"
echo "  EXP-5 COMPLETE"
echo "============================================"
echo "Results saved in: $EXP_DIR/"
echo "Cross-domain plot: plots/plot5_cross_domain.png"
