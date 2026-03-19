#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=dpo_lam0
#SBATCH --output=logs/dpo_lam0_%j.out
#SBATCH --error=logs/dpo_lam0_%j.err

# DPO Experiment — Step 2: Standard DPO (lambda=0.0)
# A100 80GB: target ~18GB + reference 8-bit ~9GB = ~27GB (no draft needed)

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp_dpo

echo ">>> Standard DPO training (lambda=0.0) ($(date))"

if [ ! -d "results/exp_dpo/baseline/final" ]; then
    python -m src.train_dpo \
        --config configs/exp_dpo_baseline.yaml \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "Training done ($(date))"
else
    echo "Already trained, skipping."
fi

echo ">>> Measuring acceptance after standard DPO"
if [ ! -f "results/exp_dpo/baseline_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "meta-llama/Llama-3.1-8B-Instruct" \
        --draft_model "meta-llama/Llama-3.2-1B-Instruct" \
        --adapter_path results/exp_dpo/baseline/final \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/baseline_acceptance_chat.json \
        --target_device cuda:0 \
        --draft_device cuda:0
fi

echo "Done ($(date))"
