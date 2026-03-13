#!/bin/bash
# EXP-7: Complementarity with Runtime Adaptation
# Shows that spec-aware FT gives adaptive systems (like ATLAS) a better
# starting point for draft model adaptation.
#
# Steps:
#   1. Generate outputs from standard-FT and spec-aware-FT code models
#   2. Fine-tune the draft model on those outputs at different steps
#   3. Measure alpha progression for both pipelines
#   4. Generate complementarity line chart
#
# Depends on: EXP-1 (standard-FT model), EXP-3 (spec-aware-FT model)
#
# Usage:
#   bash scripts/run_exp7.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  EXP-7: Complementarity with Adaptation"
echo "============================================"
echo "Project directory: $PROJECT_DIR"
echo ""

# --- Detect device ---
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
    echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    if [ "$GPU_COUNT" -ge 2 ]; then
        DRAFT_DEVICE="cuda:1"
        echo "Multi-GPU mode: target on cuda:0, draft on cuda:1"
    else
        DRAFT_DEVICE="cuda:0"
        echo "Single-GPU mode: both models on cuda:0"
    fi
else
    DEVICE="cpu"
    DRAFT_DEVICE="cpu"
    echo "No GPU detected -- running on CPU (will be slow)"
fi

# --- Extract model names from config ---
TARGET_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models']['target'])")
DRAFT_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/models.yaml'))['models']['draft'])")
echo "Target model: $TARGET_MODEL"
echo "Draft model:  $DRAFT_MODEL"
echo ""

EXP_DIR="results/exp7"
mkdir -p "$EXP_DIR"

DOMAIN="code"
STANDARD_FT_ADAPTER="results/exp1/baseline_code/final"
SPECAWARE_FT_ADAPTER="results/exp3_spec_aware_code_lam0.1/final"
ADAPTATION_STEPS="100 200 500 1000"
NUM_OUTPUTS=1000
MAX_NEW_TOKENS=256

# --- Verify prerequisites ---
echo "Checking prerequisites..."
if [ ! -d "$STANDARD_FT_ADAPTER" ]; then
    echo "ERROR: Standard-FT adapter not found at $STANDARD_FT_ADAPTER"
    echo "       Run EXP-1 first: bash scripts/run_exp1.sh"
    exit 1
fi
if [ ! -d "$SPECAWARE_FT_ADAPTER" ]; then
    echo "ERROR: Spec-aware-FT adapter not found at $SPECAWARE_FT_ADAPTER"
    echo "       Run EXP-3 first: bash scripts/run_exp3.sh"
    exit 1
fi
echo "Prerequisites met."
echo ""

# ====================================================================
# Step 1: Measure baseline alpha (before any draft adaptation)
# ====================================================================
echo ">>> Step 1/4: Measuring pre-adaptation acceptance rates"

# Standard-FT + original draft
echo "  -> Standard-FT model with original draft"
python -m src.measure_acceptance \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --adapter_path "$STANDARD_FT_ADAPTER" \
    --prompts_file configs/eval_prompts.yaml \
    --domain "$DOMAIN" \
    --output "$EXP_DIR/acceptance_standard_ft_step_0.json" \
    --max_new_tokens 128 \
    --draft_k 5 \
    --num_prompts 50 \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE"

# Spec-aware-FT + original draft
echo "  -> Spec-aware-FT model with original draft"
python -m src.measure_acceptance \
    --target_model "$TARGET_MODEL" \
    --draft_model "$DRAFT_MODEL" \
    --adapter_path "$SPECAWARE_FT_ADAPTER" \
    --prompts_file configs/eval_prompts.yaml \
    --domain "$DOMAIN" \
    --output "$EXP_DIR/acceptance_specaware_ft_step_0.json" \
    --max_new_tokens 128 \
    --draft_k 5 \
    --num_prompts 50 \
    --target_device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE"

echo ""

# ====================================================================
# Step 2: Generate outputs from both FT models
# ====================================================================
echo ">>> Step 2/4: Generating outputs for draft adaptation"

for PIPELINE in standard specaware; do
    if [ "$PIPELINE" = "standard" ]; then
        ADAPTER="$STANDARD_FT_ADAPTER"
    else
        ADAPTER="$SPECAWARE_FT_ADAPTER"
    fi

    OUTPUT_FILE="$EXP_DIR/generated_outputs_${PIPELINE}.jsonl"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "  -> $PIPELINE outputs already exist, skipping generation"
        continue
    fi

    echo "  -> Generating $NUM_OUTPUTS outputs from $PIPELINE-FT model"
    python -c "
import json
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load eval prompts for generation seeds
with open('configs/eval_prompts.yaml') as f:
    prompts_data = yaml.safe_load(f)
code_prompts = prompts_data.get('code_prompts', [])

# Load target model with adapter
target_model_name = '$TARGET_MODEL'
adapter_path = '$ADAPTER'
device = '$DEVICE'

tokenizer = AutoTokenizer.from_pretrained(target_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f'  Loading target model {target_model_name} on {device}', flush=True)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    target_model_name, torch_dtype=dtype,
).to(device)
model = PeftModel.from_pretrained(model, adapter_path).to(device)
model.eval()
print(f'  Model loaded successfully', flush=True)

# Generate outputs (cycle through prompts)
output_file = '$OUTPUT_FILE'
num_outputs = $NUM_OUTPUTS
max_new_tokens = $MAX_NEW_TOKENS

with open(output_file, 'w') as f:
    for i in range(num_outputs):
        prompt = code_prompts[i % len(code_prompts)]
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        f.write(json.dumps({'text': text, 'prompt': prompt}) + '\n')

        if (i + 1) % 100 == 0:
            print(f'  Generated {i + 1}/{num_outputs} outputs', flush=True)

print(f'  Saved {num_outputs} outputs to {output_file}', flush=True)
"
done

echo ""

# ====================================================================
# Step 3: Fine-tune draft model on generated outputs at different steps
# ====================================================================
echo ">>> Step 3/4: Adapting draft model and measuring alpha progression"

for PIPELINE in standard specaware; do
    echo "  Pipeline: $PIPELINE-FT"
    OUTPUT_FILE="$EXP_DIR/generated_outputs_${PIPELINE}.jsonl"

    if [ "$PIPELINE" = "standard" ]; then
        TARGET_ADAPTER="$STANDARD_FT_ADAPTER"
    else
        TARGET_ADAPTER="$SPECAWARE_FT_ADAPTER"
    fi

    for STEPS in $ADAPTATION_STEPS; do
        DRAFT_ADAPTER_DIR="$EXP_DIR/draft_adapted_${PIPELINE}_steps_${STEPS}"

        if [ -f "$EXP_DIR/acceptance_${PIPELINE}_ft_step_${STEPS}.json" ]; then
            echo "    -> Steps=$STEPS: already measured, skipping"
            continue
        fi

        echo "    -> Adapting draft model for $STEPS steps"

        # Fine-tune draft on generated outputs using a lightweight training run
        python -c "
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

device = torch.device('$DRAFT_DEVICE')
draft_model_name = '$DRAFT_MODEL'
output_file = '$OUTPUT_FILE'
save_dir = '$DRAFT_ADAPTER_DIR'
max_steps = $STEPS
lr = 5e-5
batch_size = 8

print(f'    Loading draft model {draft_model_name} on {device}', flush=True)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    draft_model_name, torch_dtype=dtype,
).to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
model.train()
print(f'    Model loaded, trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)

# Load generated texts
texts = []
with open(output_file) as f:
    for line in f:
        texts.append(json.loads(line)['text'])
print(f'    Loaded {len(texts)} texts from {output_file}', flush=True)

optimizer = AdamW(model.parameters(), lr=lr)
step = 0
epoch = 0

while step < max_steps:
    epoch += 1
    for i in range(0, len(texts), batch_size):
        if step >= max_steps:
            break
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors='pt', truncation=True,
            max_length=512, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        inputs['labels'][inputs['attention_mask'] == 0] = -100

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 10 == 0:
            print(f'    Draft adaptation step {step}/{max_steps}, loss={loss.item():.4f}', flush=True)

# Save adapted draft
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
print(f'    Saved adapted draft to {save_dir}', flush=True)
"

        # Measure acceptance with adapted draft
        echo "    -> Measuring alpha with adapted draft (steps=$STEPS)"
        python -c "
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

# This script loads the target model (with FT adapter) and the adapted draft
# model (with its LoRA adapter), then runs acceptance measurement.

import sys
sys.path.insert(0, '.')
from src.measure_acceptance import measure_acceptance_rate

target_model_name = '$TARGET_MODEL'
draft_model_name = '$DRAFT_MODEL'
target_adapter = '$TARGET_ADAPTER'
draft_adapter = '$DRAFT_ADAPTER_DIR'
target_device = torch.device('$DEVICE')
draft_device = torch.device('$DRAFT_DEVICE')

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load target with FT adapter
print(f'    Loading target model on {target_device}', flush=True)
tokenizer = AutoTokenizer.from_pretrained(target_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name, torch_dtype=dtype,
).to(target_device)
target_model = PeftModel.from_pretrained(target_model, target_adapter).to(target_device)
target_model.eval()
print(f'    Target model loaded', flush=True)

# Load draft with adaptation adapter
print(f'    Loading adapted draft model on {draft_device}', flush=True)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name, torch_dtype=dtype,
).to(draft_device)
draft_model = PeftModel.from_pretrained(draft_model, draft_adapter).to(draft_device)
draft_model.eval()
for p in draft_model.parameters():
    p.requires_grad = False
print(f'    Draft model loaded', flush=True)

# Load prompts
with open('configs/eval_prompts.yaml') as f:
    prompts_data = yaml.safe_load(f)
prompts = prompts_data.get('code_prompts', [])[:50]

results = measure_acceptance_rate(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    eval_prompts=prompts,
    target_device=target_device,
    draft_device=draft_device,
    max_new_tokens=128,
    draft_k=5,
)

results['pipeline'] = '$PIPELINE'
results['adaptation_steps'] = $STEPS
results['target_adapter'] = target_adapter
results['draft_adapter'] = draft_adapter

output_path = '$EXP_DIR/acceptance_${PIPELINE}_ft_step_${STEPS}.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'    alpha={results[\"alpha\"]:.4f} at step $STEPS', flush=True)
"
        echo ""
    done
done

echo ">>> Draft adaptation complete."
echo ""

# ====================================================================
# Step 4: Generate complementarity plot
# ====================================================================
echo ">>> Step 4/4: Generating plot7_complementarity.png"
python -m src.analyze_results \
    --results_dir results \
    --output_dir plots

# Print summary
python -c "
import json, os

exp_dir = '$EXP_DIR'
steps = [0] + [int(s) for s in '$ADAPTATION_STEPS'.split()]

print('COMPLEMENTARITY RESULTS')
print('=' * 70)
print(f'{\"Adaptation Steps\":>18s}  {\"Standard-FT Alpha\":>18s}  {\"Spec-Aware-FT Alpha\":>20s}')
print('-' * 70)

for s in steps:
    std_file = os.path.join(exp_dir, f'acceptance_standard_ft_step_{s}.json')
    spec_file = os.path.join(exp_dir, f'acceptance_specaware_ft_step_{s}.json')

    std_alpha = '--'
    spec_alpha = '--'

    if os.path.exists(std_file):
        d = json.load(open(std_file))
        std_alpha = f'{d[\"alpha\"]:.4f}'
    if os.path.exists(spec_file):
        d = json.load(open(spec_file))
        spec_alpha = f'{d[\"alpha\"]:.4f}'

    print(f'{s:>18d}  {std_alpha:>18s}  {spec_alpha:>20s}')

print('=' * 70)
"

echo ""
echo "============================================"
echo "  EXP-7 COMPLETE"
echo "============================================"
echo "Results saved in: $EXP_DIR/"
echo "Complementarity plot: plots/plot7_complementarity.png"
