#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=specaware_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# === Environment Setup ===
module load cuda/12.1
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# === Run Command ===
# Single-GPU mode: both models on cuda:0, draft in 4-bit
# Pass additional arguments after the script: sbatch slurm_singlegpu.sh --config configs/exp1_baseline.yaml
python src/train.py \
    --target_device cuda:0 \
    --draft_device cuda:0 \
    "$@"
