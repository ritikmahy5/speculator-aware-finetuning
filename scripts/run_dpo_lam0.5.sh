#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=dpo_lam05
#SBATCH --output=logs/dpo_lam05_%j.out
#SBATCH --error=logs/dpo_lam05_%j.err

# DPO Experiment — Step 4: Spec-aware DPO (lambda=0.5)
# A100 80GB: target ~18GB + reference 8-bit ~9GB + draft ~2GB = ~29GB

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

mkdir -p logs results/exp_dpo

echo ">>> Spec-aware DPO (lambda=0.5) ($(date))"

if [ ! -d "results/exp_dpo/specaware_lam0.5/final" ]; then
    python -m src.train_dpo \
        --config configs/exp_dpo_specaware.yaml \
        --target_device cuda:0 \
        --draft_device cuda:0
    echo "Training done ($(date))"
else
    echo "Already trained, skipping."
fi

echo ">>> Measuring acceptance after spec-aware DPO (lambda=0.5)"
if [ ! -f "results/exp_dpo/specaware_lam0.5_acceptance_chat.json" ]; then
    python -m src.measure_acceptance \
        --target_model "meta-llama/Llama-3.1-8B-Instruct" \
        --draft_model "meta-llama/Llama-3.2-1B-Instruct" \
        --adapter_path results/exp_dpo/specaware_lam0.5/final \
        --prompts_file configs/eval_prompts.yaml \
        --domain chat \
        --output results/exp_dpo/specaware_lam0.5_acceptance_chat.json \
        --target_device cuda:0 \
        --draft_device cuda:0
fi

echo "Done ($(date))"
