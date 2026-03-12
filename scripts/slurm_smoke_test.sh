#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --job-name=specaware_smoke
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# Smoke test on Explorer cluster
module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

bash scripts/smoke_test.sh
