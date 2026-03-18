#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=bench_llama
#SBATCH --output=logs/bench_llama_%j.out
#SBATCH --error=logs/bench_llama_%j.err

# Standardized benchmark evaluation for Llama adapters
# HumanEval (code), MedQA (medical), MMLU (general/chat proxy)
# Checkpoints: base, std FT, spec-aware λ=0.5, spec-aware λ=1.0

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

DEVICE="cuda:0"
TARGET="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="results/bench_eval"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  Llama Standardized Benchmark Evaluation"
echo "============================================"

# --- HumanEval (code domain) ---
echo ""
echo ">>> HumanEval (code domain)"

declare -A CODE_ADAPTERS=(
    ["base"]=""
    ["std_ft"]="results/exp1_llama/baseline_code/final"
    ["specaware_lam0.5"]="results/exp4_llama/code_lam_0.5/final"
    ["specaware_lam1.0"]="results/exp4_llama/code_lam_1.0/final"
)

for NAME in base std_ft specaware_lam0.5 specaware_lam1.0; do
    OUTF="$OUT_DIR/llama_${NAME}_humaneval.json"
    if [ -f "$OUTF" ]; then
        echo "  -> $NAME: already done"
        continue
    fi
    echo "  -> $NAME"
    ADAPTER="${CODE_ADAPTERS[$NAME]}"
    ADAPTER_ARG=""
    if [ -n "$ADAPTER" ]; then
        ADAPTER_ARG="--adapter_path $ADAPTER"
    fi
    python -m src.eval_task \
        --target_model "$TARGET" \
        $ADAPTER_ARG \
        --tasks humaneval \
        --output "$OUTF" \
        --target_device "$DEVICE" \
        --batch_size 1
done

# --- MedQA (medical domain) ---
echo ""
echo ">>> MedQA (medical domain)"

declare -A MED_ADAPTERS=(
    ["base"]=""
    ["std_ft"]="results/exp1_llama/baseline_medical/final"
    ["specaware_lam0.5"]="results/exp4_llama/medical_lam_0.5/final"
    ["specaware_lam1.0"]="results/exp4_llama/medical_lam_1.0/final"
)

for NAME in base std_ft specaware_lam0.5 specaware_lam1.0; do
    OUTF="$OUT_DIR/llama_${NAME}_medqa.json"
    if [ -f "$OUTF" ]; then
        echo "  -> $NAME: already done"
        continue
    fi
    echo "  -> $NAME"
    ADAPTER="${MED_ADAPTERS[$NAME]}"
    ADAPTER_ARG=""
    if [ -n "$ADAPTER" ]; then
        ADAPTER_ARG="--adapter_path $ADAPTER"
    fi
    python -m src.eval_task \
        --target_model "$TARGET" \
        $ADAPTER_ARG \
        --tasks medqa_4options \
        --output "$OUTF" \
        --target_device "$DEVICE" \
        --batch_size 4
done

# --- MMLU (general knowledge — chat/instruction-following proxy) ---
echo ""
echo ">>> MMLU (general knowledge)"

declare -A CHAT_ADAPTERS=(
    ["base"]=""
    ["std_ft"]="results/exp1_llama/baseline_chat/final"
    ["specaware_lam0.5"]="results/exp4_llama/chat_lam_0.5/final"
    ["specaware_lam1.0"]="results/exp4_llama/chat_lam_1.0/final"
)

for NAME in base std_ft specaware_lam0.5 specaware_lam1.0; do
    OUTF="$OUT_DIR/llama_${NAME}_mmlu.json"
    if [ -f "$OUTF" ]; then
        echo "  -> $NAME: already done"
        continue
    fi
    echo "  -> $NAME"
    ADAPTER="${CHAT_ADAPTERS[$NAME]}"
    ADAPTER_ARG=""
    if [ -n "$ADAPTER" ]; then
        ADAPTER_ARG="--adapter_path $ADAPTER"
    fi
    python -m src.eval_task \
        --target_model "$TARGET" \
        $ADAPTER_ARG \
        --tasks mmlu \
        --output "$OUTF" \
        --target_device "$DEVICE" \
        --batch_size 4
done

# Summary
echo ""
echo "============================================"
echo "  Llama Benchmark Evaluation COMPLETE"
echo "============================================"
python -c "
import json, os, glob
out_dir = '$OUT_DIR'
print(f\"{'Checkpoint':25s} {'Benchmark':15s} {'Score':>10s}\")
print('-' * 55)
for f in sorted(glob.glob(os.path.join(out_dir, 'llama_*.json'))):
    name = os.path.basename(f).replace('.json', '')
    data = json.load(open(f))
    lm = data.get('lm_eval', {})
    for task, metrics in lm.items():
        # Find the main metric
        score = metrics.get('acc,none', metrics.get('acc_norm,none', metrics.get('pass@1', 'N/A')))
        print(f'{name:25s} {task:15s} {score:>10s}' if isinstance(score, str) else f'{name:25s} {task:15s} {score:10.4f}')
"
