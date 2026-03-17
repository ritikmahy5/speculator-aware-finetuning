#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --job-name=exp1_llama
#SBATCH --output=logs/exp1_llama_%j.out
#SBATCH --error=logs/exp1_llama_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mahyavanshiritik2311@gmail.com

module load cuda/12.1
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp1_llama

echo "=== EXP-1 LLAMA: Starting $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

# -------------------------------------------------------
# Phase 1: Measure base acceptance rates and KL
# -------------------------------------------------------
echo "--- Phase 1: Base model measurements ---"
for domain in code medical chat mixed; do
    echo "Measuring base acceptance: $domain"
    python src/measure_acceptance.py \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --draft_model meta-llama/Llama-3.1-1B-Instruct \
        --domain $domain \
        --target_device cuda:0 --draft_device cuda:1 \
        --output results/exp1_llama/acceptance_base_${domain}.json

    echo "Measuring base KL: $domain"
    python src/measure_kl.py \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --draft_model meta-llama/Llama-3.1-1B-Instruct \
        --domain $domain \
        --target_device cuda:0 --draft_device cuda:1 \
        --output results/exp1_llama/kl_base_${domain}.json
done

# -------------------------------------------------------
# Phase 2: Fine-tune on each domain and measure
# -------------------------------------------------------
for domain in code medical chat; do
    echo "--- Phase 2: Fine-tuning on $domain ---"
    python src/train.py \
        --config configs/exp1_baseline_llama.yaml \
        --domain $domain \
        --target_device cuda:0 --draft_device cuda:1 \
        --output_dir results/exp1_llama/${domain}_baseline

    echo "--- Phase 2: Measuring FT model on all domains ---"
    for eval_domain in code medical chat mixed; do
        echo "Measuring acceptance: FT=$domain, Eval=$eval_domain"
        python src/measure_acceptance.py \
            --target_model meta-llama/Llama-3.1-8B-Instruct \
            --draft_model meta-llama/Llama-3.1-1B-Instruct \
            --adapter_path results/exp1_llama/${domain}_baseline/final \
            --domain $eval_domain \
            --target_device cuda:0 --draft_device cuda:1 \
            --output results/exp1_llama/acceptance_${domain}_baseline_eval_${eval_domain}.json

        echo "Measuring KL: FT=$domain, Eval=$eval_domain"
        python src/measure_kl.py \
            --target_model meta-llama/Llama-3.1-8B-Instruct \
            --draft_model meta-llama/Llama-3.1-1B-Instruct \
            --adapter_path results/exp1_llama/${domain}_baseline/final \
            --domain $eval_domain \
            --target_device cuda:0 --draft_device cuda:1 \
            --output results/exp1_llama/kl_${domain}_baseline_eval_${eval_domain}.json
    done
done

echo "=== EXP-1 LLAMA: Complete $(date) ==="
