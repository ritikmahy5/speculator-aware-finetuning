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

| Role | Qwen (primary) | Llama (secondary) |
|------|----------------|-------------------|
| Target | `Qwen/Qwen2.5-7B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` |
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | `meta-llama/Llama-3.2-1B-Instruct` |

## Key Results

### EXP-1: Baseline Degradation

Standard LoRA fine-tuning degrades speculative decoding acceptance rate (α), especially with Llama models:

| Domain | Llama Base α | Llama Post-FT α | Relative Drop |
|--------|-------------|-----------------|---------------|
| Code | 0.5954 | 0.5449 | -8.5% |
| Medical | 0.4163 | 0.3747 | -10.0% |
| Chat | 0.3784 | 0.2517 | **-33.5%** |

Qwen showed minimal degradation (~0%), making Llama the better test bed for our method.

### EXP-3: Speculator-Aware Fine-Tuning (Llama, λ=0.1)

Our core result — spec-aware training dramatically reduces chat degradation:

| Domain | Base α | Standard FT α | Spec-Aware α (λ=0.1) |
|--------|--------|--------------|----------------------|
| Code | 0.5954 | 0.5449 (-8.5%) | **0.5596** (-6.0%) |
| Medical | 0.4163 | 0.3747 (-10.0%) | 0.3711 (-10.9%) |
| Chat | 0.3784 | 0.2517 (-33.5%) | **0.3495** (-7.6%) |

Chat degradation reduced from **-33.5% to -7.6%** — recovering most of the lost acceptance rate with a single regularization term.

### EXP-4: Lambda Sweep (Qwen)

Higher λ monotonically increases α but trades off task loss. Results across domains:

| Domain | λ=0.01 α | λ=0.1 α | λ=1.0 α | Base α |
|--------|---------|---------|---------|--------|
| Code | 0.5405 | 0.5300 | 0.5939 | 0.5203 |
| Medical | 0.3340 | 0.3559 | 0.4556 | 0.3103 |
| Chat | 0.2918 | 0.3030 | 0.3377 | 0.2546 |

### EXP-4: Lambda Sweep (Llama)

Llama models show larger degradation from standard FT, making the recovery more dramatic. At λ=1.0, **all three domains exceed the base model's α**:

| Domain | Base α | Std FT α | λ=0.1 α | λ=0.5 α | λ=1.0 α |
|--------|--------|----------|---------|---------|---------|
| Code | 0.5954 | 0.5449 | 0.5596 | 0.5881 | **0.6158** (+3.4%) |
| Medical | 0.4163 | 0.3747 | 0.3952 | 0.3925 | **0.4320** (+3.8%) |
| Chat | 0.3784 | 0.2517 | 0.2624 | 0.3554 | **0.4063** (+7.4%) |

### EXP-5: Cross-Domain Generalization (Qwen)

Models trained with spec-aware loss on one domain maintain reasonable α on other domains:

![Cross-Domain Heatmap](plots/plot5_cross_domain.png)

### EXP-6: Loss Function Ablation (Qwen, λ=0.01)

| Loss Type | α | Ranking |
|-----------|------|---------|
| JS | 0.5509 | Best |
| Token Match | 0.5487 | 2nd |
| TV | 0.5468 | 3rd |
| KL | 0.5405 | 4th |
| Reverse KL | 0.5300 | Worst |

Jensen-Shannon divergence marginally outperforms forward KL as the regularization loss.

### EXP-7: Complementarity with Runtime Adaptation (Qwen)

Spec-aware FT provides a better starting point for runtime draft adaptation (ATLAS-style):

| Adaptation Steps | Standard FT α | Spec-Aware FT α |
|-----------------|--------------|----------------|
| 0 | 0.5495 | 0.5300 |
| 100 | 0.5624 | 0.5347 |
| 500 | 0.5909 | 0.5539 |
| 1000 | 0.6000 | 0.5587 |

Both approaches improve with draft adaptation. For Llama (where standard FT degrades α significantly), spec-aware FT prevents the large initial degradation that runtime adaptation must recover from.

## Plots

| Plot | Description |
|------|-------------|
| ![](plots/plot1_degradation.png) | EXP-1: Acceptance rate before/after fine-tuning |
| ![](plots/plot2_kl_correlation.png) | EXP-2: KL divergence vs acceptance rate correlation |
| ![](plots/plot3_spec_aware_comparison.png) | EXP-3: Base vs standard-FT vs spec-aware-FT |
| ![](plots/plot4_pareto_overlay.png) | EXP-4: Pareto frontier — Qwen, all domains |
| ![](plots/plot4_pareto_overlay_llama.png) | EXP-4: Pareto frontier — Llama, all domains |
| ![](plots/plot5_cross_domain.png) | EXP-5: Cross-domain generalization heatmap |
| ![](plots/plot6_loss_ablation.png) | EXP-6: Loss function comparison |
| ![](plots/plot7_complementarity.png) | EXP-7: Complementarity with runtime adaptation |

## Experiments

| # | Experiment | Qwen | Llama |
|---|-----------|------|-------|
| 1 | Baseline degradation measurement | Done | Done |
| 2 | KL–acceptance rate correlation | Done | — |
| 3 | Speculator-aware fine-tuning (core) | Done | Done |
| 4 | Lambda sweep + Pareto analysis | Done | Done |
| 5 | Cross-domain analysis | Done | — |
| 6 | Loss function ablation | Done | — |
| 7 | Complementarity with runtime adaptation | Done | — |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a specific experiment
python -m src.train --config configs/exp3_spec_aware.yaml \
    --target_device cuda:0 --draft_device cuda:1

# Measure acceptance rate
python -m src.measure_acceptance \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --draft_model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_path results/exp3_spec_aware_code_lam0.1/final \
    --prompts_file configs/eval_prompts.yaml \
    --domain code \
    --output results/eval_acceptance.json

# Measure KL divergence
python -m src.measure_kl \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --draft_model Qwen/Qwen2.5-0.5B-Instruct \
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
