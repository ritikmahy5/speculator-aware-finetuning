#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp3_llama
#SBATCH --output=logs/exp3_llama_%j.out
#SBATCH --error=logs/exp3_llama_%j.err

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# Swap to Llama models config
cp configs/models.yaml configs/models_qwen_backup.yaml
cp configs/models_llama.yaml configs/models.yaml

# Override results directory and training config for Llama
export EXP3_RESULTS_DIR="results/exp3_llama_spec_aware_code_lam0.1"
export EXP3_CONFIG="configs/exp3_spec_aware_llama.yaml"
export EXP1_RESULTS_DIR="results/exp1_llama"

bash scripts/run_exp3.sh

# Restore Qwen config
cp configs/models_qwen_backup.yaml configs/models.yaml
rm configs/models_qwen_backup.yaml
