#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=stress_test
#SBATCH --output=logs/stress_test_%j.out
#SBATCH --error=logs/stress_test_%j.err

# Stress Test: High-rank (64), 3-epoch Qwen fine-tuning
# Tests whether Qwen can be pushed to show α degradation with more aggressive training.
# Saves checkpoints at end of each epoch for tracking degradation trajectory.

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

EXP_DIR="results/stress_test_qwen"
mkdir -p "$EXP_DIR"

DEVICE="cuda:0"
TARGET="Qwen/Qwen2.5-7B-Instruct"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct"

echo "============================================"
echo "  Stress Test: Qwen rank=64, 3 epochs"
echo "============================================"

# Step 1: Measure base α for reference
echo ">>> Measuring base model α"
python -m src.measure_acceptance \
    --target_model "$TARGET" \
    --draft_model "$DRAFT" \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output "$EXP_DIR/acceptance_base.json" \
    --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
    --target_device "$DEVICE" --draft_device "$DEVICE"

# Step 2: Fine-tune with rank=64, 3 epochs
# 10K samples, batch=4, grad_accum=4 -> 625 steps/epoch, 1875 total
# Save checkpoint every 468 steps (~75% of each epoch) to get 4 checkpoints
echo ">>> Fine-tuning: rank=64, 3 epochs, code domain"
python -m src.train \
    --config configs/stress_test_qwen.yaml \
    --target_device "$DEVICE" \
    --draft_device "$DEVICE" \
    --output_dir "$EXP_DIR/rank64_3epoch"

# Step 3: Measure α at each checkpoint and final
echo ">>> Measuring α at checkpoints"

CKPT_DIR="$EXP_DIR/rank64_3epoch/checkpoints"
FINAL_DIR="$EXP_DIR/rank64_3epoch/final"
CHECKPOINTS=""
if [ -d "$CKPT_DIR" ]; then
    CHECKPOINTS=$(ls -d "$CKPT_DIR"/step_* 2>/dev/null | sort -t_ -k2 -n || true)
fi
if [ -d "$FINAL_DIR" ]; then
    CHECKPOINTS="$CHECKPOINTS $FINAL_DIR"
fi

for CKPT in $CHECKPOINTS; do
    CKPT_NAME=$(basename "$CKPT")
    echo "  -> Checkpoint: $CKPT_NAME"

    python -m src.measure_acceptance \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path "$CKPT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$EXP_DIR/acceptance_${CKPT_NAME}.json" \
        --max_new_tokens 128 --draft_k 5 --num_prompts 50 \
        --target_device "$DEVICE" --draft_device "$DEVICE"

    python -m src.measure_kl \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --adapter_path "$CKPT" \
        --prompts_file configs/eval_prompts.yaml \
        --domain code \
        --output "$EXP_DIR/kl_${CKPT_NAME}.json" \
        --divergence_types kl js tv \
        --target_device "$DEVICE" --draft_device "$DEVICE"
done

# Summary
echo ""
echo "============================================"
echo "  Stress Test COMPLETE"
echo "============================================"
python -c "
import json, glob, os
exp_dir = '$EXP_DIR'
base = json.load(open(os.path.join(exp_dir, 'acceptance_base.json')))
base_alpha = base.get('alpha', base.get('mean_alpha', 0))
print(f'Base α: {base_alpha:.4f}')
print()
for f in sorted(glob.glob(os.path.join(exp_dir, 'acceptance_step_*.json')) + glob.glob(os.path.join(exp_dir, 'acceptance_final.json'))):
    name = os.path.basename(f).replace('acceptance_', '').replace('.json', '')
    data = json.load(open(f))
    a = data.get('alpha', data.get('mean_alpha', 0))
    delta = (a - base_alpha) / base_alpha * 100
    print(f'{name:15s}  α={a:.4f}  ({delta:+.1f}% vs base)')
"
