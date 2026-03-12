#!/bin/bash
# Smoke test: verify the full pipeline works end-to-end with tiny data.
# Should complete in <10 minutes on 1 GPU.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Speculator-Aware FT — Smoke Test"
echo "============================================"

# Determine device
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
    echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    DEVICE="cpu"
    echo "No GPU detected — running on CPU (will be slow)"
fi

SMOKE_DIR="results/smoke_test"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

# ---- Step 1: Baseline training (lam=0.0) ----
echo ""
echo ">>> Step 1/5: Baseline training (lam=0.0, 100 samples, 1 epoch)"
python -m src.train \
    --config configs/exp1_baseline.yaml \
    --domain code \
    --target_device "$DEVICE" \
    --draft_device "$DEVICE" \
    --training.max_samples 100 \
    --training.num_epochs 1 \
    --training.batch_size 2 \
    --training.gradient_accumulation_steps 1 \
    --training.max_seq_len 256 \
    --spec_loss.lam 0.0 \
    --output_dir "$SMOKE_DIR/baseline" \
    --logging.log_every 5

echo ">>> Baseline training complete."

# ---- Step 2: Measure acceptance rate on baseline ----
echo ""
echo ">>> Step 2/5: Measuring acceptance rate on baseline model"
python -m src.measure_acceptance \
    --target_model "$(cat configs/models.yaml | python -c "import sys,yaml; print(yaml.safe_load(sys.stdin)['models']['target'])")" \
    --draft_model "$(cat configs/models.yaml | python -c "import sys,yaml; print(yaml.safe_load(sys.stdin)['models']['draft'])")" \
    --adapter_path "$SMOKE_DIR/baseline/final" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$SMOKE_DIR/acceptance_baseline.json" \
    --max_new_tokens 32 \
    --draft_k 3 \
    --num_prompts 5 \
    --target_device "$DEVICE" \
    --draft_device "$DEVICE"

echo ">>> Baseline acceptance measurement complete."

# ---- Step 3: Spec-aware training (lam=0.1) ----
echo ""
echo ">>> Step 3/5: Spec-aware training (lam=0.1, 100 samples, 1 epoch)"
python -m src.train \
    --config configs/exp3_spec_aware.yaml \
    --domain code \
    --target_device "$DEVICE" \
    --draft_device "$DEVICE" \
    --training.max_samples 100 \
    --training.num_epochs 1 \
    --training.batch_size 2 \
    --training.gradient_accumulation_steps 1 \
    --training.max_seq_len 256 \
    --spec_loss.lam 0.1 \
    --output_dir "$SMOKE_DIR/spec_aware" \
    --logging.log_every 5

echo ">>> Spec-aware training complete."

# ---- Step 4: Measure acceptance rate on spec-aware ----
echo ""
echo ">>> Step 4/5: Measuring acceptance rate on spec-aware model"
python -m src.measure_acceptance \
    --target_model "$(cat configs/models.yaml | python -c "import sys,yaml; print(yaml.safe_load(sys.stdin)['models']['target'])")" \
    --draft_model "$(cat configs/models.yaml | python -c "import sys,yaml; print(yaml.safe_load(sys.stdin)['models']['draft'])")" \
    --adapter_path "$SMOKE_DIR/spec_aware/final" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$SMOKE_DIR/acceptance_spec_aware.json" \
    --max_new_tokens 32 \
    --draft_k 3 \
    --num_prompts 5 \
    --target_device "$DEVICE" \
    --draft_device "$DEVICE"

echo ">>> Spec-aware acceptance measurement complete."

# ---- Step 5: Compare results ----
echo ""
echo ">>> Step 5/5: Comparing results"
python -c "
import json

with open('$SMOKE_DIR/acceptance_baseline.json') as f:
    baseline = json.load(f)
with open('$SMOKE_DIR/acceptance_spec_aware.json') as f:
    spec_aware = json.load(f)

print('=' * 50)
print('SMOKE TEST RESULTS')
print('=' * 50)
print(f'Baseline alpha:    {baseline[\"alpha\"]:.4f}')
print(f'Spec-aware alpha:  {spec_aware[\"alpha\"]:.4f}')
diff = spec_aware['alpha'] - baseline['alpha']
print(f'Difference:        {diff:+.4f}')
print()
print('Pipeline verification: PASSED')
print('(Note: with only 100 training samples and 5 eval prompts,')
print(' these numbers are NOT meaningful — only verifying the pipeline works)')
print('=' * 50)
"
echo ""
echo "Smoke test complete!"
