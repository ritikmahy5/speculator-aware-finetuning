#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --job-name=specaware_multi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# === Environment Setup ===
module load cuda/12.1.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface
cd /scratch/$USER/speculator-aware-finetuning

# === Run Command ===
# Two-GPU mode: target on cuda:0, draft on cuda:1
# Pass additional arguments after the script: sbatch slurm_multigpu.sh --config configs/exp3_spec_aware.yaml
python src/train.py \
    --target_device cuda:0 \
    --draft_device cuda:1 \
    "$@"
