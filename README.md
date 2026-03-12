# Speculator-Aware Fine-Tuning

**Can we fine-tune LLMs without breaking speculative decoding?**

When a target model is fine-tuned on domain-specific data, its output distribution shifts away from the draft model used for speculative decoding — degrading inference speed. This project investigates whether adding a KL-divergence regularization loss during LoRA fine-tuning can preserve speculative decoding acceptance rates.

## The Approach

```
L_total = L_task + λ × L_spec
L_task  = CrossEntropy(target_model(x), y)       # standard task loss
L_spec  = KL(p_target_finetuned || p_draft)       # speculator alignment loss
```

The draft model is frozen and loaded alongside the target during training. It provides logits for the regularization term — no changes to model architecture or inference code required.

## Models

| Role | Model | Size |
|------|-------|------|
| Target | `meta-llama/Llama-3.1-8B-Instruct` | 8B |
| Draft | `meta-llama/Llama-3.2-1B-Instruct` | 1B |

## Experiments

| # | Experiment | Status |
|---|-----------|--------|
| 1 | Baseline degradation measurement | Pending |
| 2 | KL–acceptance rate correlation | Pending |
| 3 | Speculator-aware fine-tuning (core) | Pending |
| 4 | Lambda sweep + Pareto analysis | Pending |
| 5 | Cross-domain analysis | Pending |
| 6 | Loss function ablation | Pending |
| 7 | Complementarity with runtime adaptation | Pending |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test (verifies full pipeline)
bash scripts/smoke_test.sh

# Run a specific experiment
python -m src.train --config configs/exp3_spec_aware.yaml \
    --target_device cuda:0 --draft_device cuda:1

# Measure acceptance rate
python -m src.measure_acceptance \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path results/exp3_spec_aware_code_lam0.1/final \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output results/eval_acceptance.json

# Measure KL divergence
python -m src.measure_kl \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path results/exp3_spec_aware_code_lam0.1/final \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output results/eval_kl.json
```

## Repository Structure

```
├── src/
│   ├── train.py               # Training loop with LoRA + spec loss
│   ├── spec_loss.py           # All loss variants (KL, JS, TV, reverse KL, token-match)
│   ├── measure_acceptance.py  # Speculative decoding acceptance rate measurement
│   ├── measure_kl.py          # KL/JS/TV divergence measurement
│   ├── data.py                # Dataset loading and preprocessing
│   ├── utils.py               # Config loading, logging, device detection
│   └── analyze_results.py     # Plotting and results analysis
├── configs/                   # Experiment configs (YAML)
├── scripts/                   # Shell scripts for running experiments
├── results/                   # Experiment outputs (JSON metrics, logs)
├── plots/                     # Generated figures
└── docs/                      # Research plan and report
```

## Loss Types

| Type | Formula | Description |
|------|---------|-------------|
| `kl` | KL(target ‖ draft) | Forward KL — penalizes target mass where draft has none |
| `reverse_kl` | KL(draft ‖ target) | Mode-seeking — penalizes not covering draft peaks |
| `js` | JS(target, draft) | Symmetric, bounded Jensen-Shannon divergence |
| `tv` | 0.5 × Σ\|p-q\| | Total Variation — directly related to acceptance rate |
| `token_match` | 1 - P(argmax match) | Fraction where top-1 tokens differ |

## License

MIT
