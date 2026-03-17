#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp6_llama
#SBATCH --output=logs/exp6_llama_%j.out
#SBATCH --error=logs/exp6_llama_%j.err

# EXP-6 Llama: Loss Function Ablation at λ=0.5
# Tests all 5 loss variants on Llama code domain at λ=0.5
# (higher than Qwen's λ=0.01 to actually differentiate loss types)

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# Use Llama models
cp configs/models_llama.yaml configs/models.yaml

# Run EXP-6 with λ=0.5 on Llama
export EXP6_RESULTS_DIR="results/exp6_llama"
export EXP6_CONFIG="configs/exp6_loss_ablation_llama.yaml"
bash scripts/run_exp6.sh --lam 0.5
