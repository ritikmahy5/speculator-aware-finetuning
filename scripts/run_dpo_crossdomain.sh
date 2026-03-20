#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=dpo_xdomain
#SBATCH --output=logs/dpo_xdomain_%j.out
#SBATCH --error=logs/dpo_xdomain_%j.err

# DPO Cross-Domain Evaluation
# Measures acceptance rate of chat-DPO trained models on code and medical prompts
# Base acceptance already exists from EXP-1 Llama

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp_dpo

TARGET="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="meta-llama/Llama-3.2-1B-Instruct"

# Measure each DPO model on code and medical prompts
for adapter in baseline specaware_lam0.1 specaware_lam0.5; do
    for domain in code medical; do
        outfile="results/exp_dpo/${adapter}_acceptance_${domain}.json"
        if [ -f "$outfile" ]; then
            echo ">>> Already exists: $outfile, skipping."
            continue
        fi
        echo ">>> Measuring $adapter on $domain ($(date))"
        python -m src.measure_acceptance \
            --target_model "$TARGET" \
            --draft_model "$DRAFT" \
            --adapter_path "results/exp_dpo/${adapter}/final" \
            --prompts_file configs/eval_prompts.yaml \
            --domain "$domain" \
            --output "$outfile" \
            --target_device cuda:0 \
            --draft_device cuda:0
        echo ">>> Done: $outfile ($(date))"
    done
done

echo "All DPO cross-domain measurements complete ($(date))"
