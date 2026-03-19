#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=bench_humaneval
#SBATCH --output=logs/bench_humaneval_%j.out
#SBATCH --error=logs/bench_humaneval_%j.err

# HumanEval-only rerun for both Llama and Qwen
# Previous run failed: lm-eval needed confirm_run_unsafe_code=True

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
export HF_ALLOW_CODE_EVAL=1
cd /scratch/$USER/speculator-aware-finetuning

DEVICE="cuda:0"
OUT_DIR="results/bench_eval"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  HumanEval Rerun (Llama + Qwen)"
echo "============================================"

# --- Llama HumanEval ---
echo ""
echo ">>> Llama HumanEval"
TARGET="meta-llama/Llama-3.1-8B-Instruct"

declare -A LLAMA_ADAPTERS=(
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
    ADAPTER="${LLAMA_ADAPTERS[$NAME]}"
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

# --- Qwen HumanEval ---
echo ""
echo ">>> Qwen HumanEval"
TARGET="Qwen/Qwen2.5-7B-Instruct"

declare -A QWEN_ADAPTERS=(
    ["base"]=""
    ["std_ft"]="results/exp1/baseline_code/final"
    ["specaware_lam0.5"]="results/exp4/code_lam_0.5/final"
    ["specaware_lam1.0"]="results/exp4/code_lam_1.0/final"
)

for NAME in base std_ft specaware_lam0.5 specaware_lam1.0; do
    OUTF="$OUT_DIR/qwen_${NAME}_humaneval.json"
    if [ -f "$OUTF" ]; then
        echo "  -> $NAME: already done"
        continue
    fi
    echo "  -> $NAME"
    ADAPTER="${QWEN_ADAPTERS[$NAME]}"
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

echo ""
echo "============================================"
echo "  HumanEval Rerun COMPLETE"
echo "============================================"
