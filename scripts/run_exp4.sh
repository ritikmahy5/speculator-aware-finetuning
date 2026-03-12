#!/bin/bash
# EXP-4: Lambda Sweep + Pareto Analysis
# Finds optimal lambda per domain and produces Pareto frontiers.
#
# Steps:
#   1. For each domain x lambda, fine-tune with spec-aware loss
#   2. Measure alpha and task performance for each
#   3. Generate Pareto frontier plots
#
# Depends on: EXP-3 (validates spec-aware approach works)
# Note: lam=0.0 results come from EXP-1; we skip re-running those.
#
# Usage:
#   bash scripts/run_exp4.sh
#   bash scripts/run_exp4.sh --domain code          # single domain
#   bash scripts/run_exp4.sh --lam 0.1              # single lambda
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-4: Lambda Sweep + Pareto Analysis"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# --- Parse optional CLI filters ---
FILTER_DOMAIN=""
FILTER_LAM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain) FILTER_DOMAIN="$2"; shift 2 ;;
        --lam)    FILTER_LAM="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
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

EXP_DIR="results/exp4"
mkdir -p "$EXP_DIR"

# --- Define sweep parameters ---
if [ -n "$FILTER_DOMAIN" ]; then
    DOMAINS="$FILTER_DOMAIN"
else
    DOMAINS="code medical chat"
fi

if [ -n "$FILTER_LAM" ]; then
    LAMBDAS="$FILTER_LAM"
else
    LAMBDAS="0.01 0.05 0.1 0.2 0.5 1.0"
fi

# Count total runs
TOTAL=0
for DOMAIN in $DOMAINS; do
    for LAM in $LAMBDAS; do
        TOTAL=$((TOTAL + 1))
    done
done
echo "Total runs: $TOTAL (domains: $DOMAINS | lambdas: $LAMBDAS)"
echo ""

# ====================================================================
# Step 1-2: Fine-tune and measure for each domain x lambda
# ====================================================================
RUN_NUM=0
for DOMAIN in $DOMAINS; do
    for LAM in $LAMBDAS; do
        RUN_NUM=$((RUN_NUM + 1))
        RUN_DIR="$EXP_DIR/${DOMAIN}_lam_${LAM}"

        echo ">>> Run $RUN_NUM/$TOTAL: domain=$DOMAIN, lam=$LAM"

        # Skip if results already exist
        if [ -f "$RUN_DIR/acceptance_${DOMAIN}.json" ]; then
            echo "  -> Skipping (results already exist)"
            echo ""
            continue
        fi

        mkdir -p "$RUN_DIR"

        # Fine-tune
        echo "  -> Fine-tuning..."
        python -m src.train \
            --config configs/exp4_lambda_sweep.yaml \
            --domain "$DOMAIN" \
            --target_device "$DEVICE" \
            --draft_device "$DRAFT_DEVICE" \
            --spec_loss.lam "$LAM" \
            --spec_loss.loss_type kl \
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
        echo "  -> Measuring KL divergence..."
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
done

# ====================================================================
# Step 3: Generate Pareto plots
# ====================================================================
echo ">>> Generating Pareto frontier plots"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots

# Print summary table
python -c "
import json, os

exp_dir = '$EXP_DIR'
domains = '$DOMAINS'.split()
lambdas = '$LAMBDAS'.split()

print('LAMBDA SWEEP RESULTS')
print('=' * 80)
print(f'{\"Domain\":10s} {\"Lambda\":>8s} {\"Alpha\":>10s} {\"Alpha Std\":>10s} {\"KL Mean\":>10s}')
print('-' * 80)

for domain in domains:
    for lam in lambdas:
        run_dir = os.path.join(exp_dir, f'{domain}_lam_{lam}')
        acc_file = os.path.join(run_dir, f'acceptance_{domain}.json')
        kl_file = os.path.join(run_dir, f'kl_{domain}.json')
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
            print(f'{domain:10s} {lam:>8s} {alpha:10.4f} {alpha_std:10.4f} {kl_mean:>10s}')
        else:
            print(f'{domain:10s} {lam:>8s}       (not yet run)')
    print()

print('=' * 80)
"

echo ""
echo "============================================"
echo "  EXP-4 COMPLETE"
echo "============================================"
echo "Results saved in: $EXP_DIR/"
echo "Pareto plots:     plots/plot4_pareto_*.png"
