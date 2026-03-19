#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=dpo_base
#SBATCH --output=logs/dpo_base_%j.out
#SBATCH --error=logs/dpo_base_%j.err

# DPO Experiment — Step 1: Base acceptance rate measurement

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp_dpo

echo ">>> Base acceptance rate measurement ($(date))"

if [ ! -f "results/exp_dpo/base_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "meta-llama/Llama-3.1-8B-Instruct" \
        --draft_model "meta-llama/Llama-3.2-1B-Instruct" \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/base_acceptance_chat.json \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "Done ($(date))"
else
    echo "Already exists, skipping."
fi
