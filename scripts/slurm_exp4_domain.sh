#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=exp4_sweep
#SBATCH --output=logs/exp4_%j.out
#SBATCH --error=logs/exp4_%j.err

module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# Domain passed as first argument, e.g.: sbatch scripts/slurm_exp4_domain.sh code
DOMAIN=${1:-code}
bash scripts/run_exp4.sh --domain "$DOMAIN"
