#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=download_llama
#SBATCH --output=logs/download_llama_%j.out
#SBATCH --error=logs/download_llama_%j.err

source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/specaware
export HF_HOME=/scratch/$USER/.cache/huggingface

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Llama-3.1-8B-Instruct...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('Done.')
"
