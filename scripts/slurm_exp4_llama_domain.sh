#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp4_llama
#SBATCH --output=logs/exp4_llama_%j.out
#SBATCH --error=logs/exp4_llama_%j.err

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# Swap to Llama models config
cp configs/models.yaml configs/models_qwen_backup.yaml
cp configs/models_llama.yaml configs/models.yaml

# Override for Llama
export EXP4_RESULTS_DIR="results/exp4_llama"
export EXP4_CONFIG="configs/exp4_lambda_sweep_llama.yaml"

# Domain passed as first argument, e.g.: sbatch scripts/slurm_exp4_llama_domain.sh code
DOMAIN=${1:-code}
bash scripts/run_exp4.sh --domain "$DOMAIN"

# Restore Qwen config
cp configs/models_qwen_backup.yaml configs/models.yaml
rm configs/models_qwen_backup.yaml
