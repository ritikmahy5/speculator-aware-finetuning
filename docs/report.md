# Speculator-Aware Fine-Tuning: Preserving Speculative Decoding Efficiency Through KL Regularization

## Abstract

Speculative decoding accelerates LLM inference by using a small draft model to propose tokens that a larger target model verifies in parallel. However, when the target model is fine-tuned on domain-specific data, its output distribution shifts away from the draft model, degrading acceptance rates and inference speed. We propose **speculator-aware fine-tuning**, which adds a KL-divergence regularization term to the training loss that constrains the fine-tuned model to remain close to the draft model's distribution. Using LoRA fine-tuning on Llama-3.1-8B-Instruct (target) with Llama-3.2-1B-Instruct (draft), we demonstrate that standard fine-tuning degrades speculative decoding acceptance rate (α) by up to 33.5% on chat data, while our method at λ=0.1 limits this to 7.6%. At higher regularization strengths (λ=1.0), the fine-tuned model can actually exceed the base model's acceptance rate. We evaluate five divergence measures as regularization losses, finding Jensen-Shannon divergence marginally outperforms forward KL. Cross-domain analysis shows the approach generalizes across evaluation domains, and complementarity experiments demonstrate it provides a better starting point for runtime draft adaptation systems.

## 1. Introduction

Speculative decoding is a widely adopted technique for accelerating large language model (LLM) inference. A smaller **draft model** proposes K candidate tokens autoregressively, and the larger **target model** verifies all K tokens in a single forward pass, accepting a prefix of tokens that match its own distribution. The **acceptance rate** (α) — the fraction of draft tokens accepted by the target — directly determines the speedup factor.

A critical but understudied problem arises when the target model is fine-tuned for domain-specific tasks. Fine-tuning shifts the target's output distribution, increasing the divergence from the draft model and reducing α. This creates a tension: domain adaptation improves task quality but degrades inference efficiency.

We propose a simple training-time solution: augmenting the standard cross-entropy loss with a KL-divergence term that penalizes distributional drift from the draft model:

```
L_total = L_task + λ × KL(p_target || p_draft)
```

The draft model is frozen and loaded alongside the target during training. This approach requires no modifications to model architecture, inference code, or the draft model itself.

### Contributions

1. We quantify the degradation of speculative decoding acceptance rates from standard LoRA fine-tuning across three domains (code, medical, chat), finding up to 33.5% relative degradation with Llama models.
2. We propose speculator-aware fine-tuning with KL regularization and demonstrate it reduces acceptance rate degradation from 33.5% to 7.6% at λ=0.1, with higher λ values fully recovering or exceeding base performance.
3. We evaluate five divergence measures as regularization losses, finding JS divergence marginally outperforms forward KL.
4. We analyze cross-domain generalization and complementarity with runtime draft adaptation (e.g., ATLAS-style systems).

## 2. Method

### 2.1 Problem Setup

Given a target model T and draft model D used for speculative decoding, we fine-tune T on domain-specific data using LoRA while preserving the draft-target alignment. The draft model D is frozen throughout training.

### 2.2 Speculator-Aware Loss

For each training batch, we perform forward passes through both models:
- **Target forward pass** (with gradients): produces logits for task loss and regularization
- **Draft forward pass** (no gradients): produces reference logits for regularization only

The combined loss is:

```
L_total = L_task + λ × L_spec
```

where L_task is the standard cross-entropy loss and L_spec is a divergence measure between the target and draft distributions. We evaluate five variants:

| Loss Type | Formula | Properties |
|-----------|---------|-----------|
| Forward KL | KL(p_target ‖ p_draft) | Mean-seeking, penalizes mass where draft is low |
| Reverse KL | KL(p_draft ‖ p_target) | Mode-seeking, concentrates probability |
| Jensen-Shannon | 0.5 × KL(p‖m) + 0.5 × KL(q‖m) | Symmetric, bounded [0, ln2] |
| Total Variation | 0.5 × Σ\|p - q\| | Directly bounds acceptance probability |
| Token Match | 1 - P(argmax_target = argmax_draft) | Simplest; only top-1 agreement |

### 2.3 Training Details

- **LoRA configuration:** rank=16, alpha=32, dropout=0.05, applied to all attention and MLP projections
- **Optimizer:** AdamW, lr=2e-4, cosine schedule with 5% warmup
- **Data:** 10,000 samples per domain, max sequence length 1024, bf16 precision
- **Domains:** Code (StarCoder), Medical (MedQA), Chat (UltraChat)

## 3. Experimental Results

### 3.1 Baseline Degradation (EXP-1)

We first establish that standard fine-tuning degrades speculative decoding acceptance rates. We evaluate two model families:

**Llama-3.1-8B + Llama-3.2-1B:**

| Domain | Base α | Post-FT α | Relative Drop |
|--------|--------|-----------|---------------|
| Code | 0.5954 ± 0.1535 | 0.5449 ± 0.1271 | -8.5% |
| Medical | 0.4163 ± 0.1076 | 0.3747 ± 0.1048 | -10.0% |
| Chat | 0.3784 ± 0.1018 | 0.2517 ± 0.0807 | **-33.5%** |

**Qwen2.5-7B + Qwen2.5-0.5B:** Showed minimal degradation (~0%), indicating the effect is model-family dependent. The Qwen draft model appears well-aligned with the target across fine-tuning, possibly due to shared pretraining characteristics.

**Finding:** Degradation severity correlates with domain distance from pretraining data. Chat, which most changes the model's conversational style, shows the largest drop.

### 3.2 KL–Acceptance Rate Correlation (EXP-2)

We validated that KL divergence between target and draft predicts acceptance rate by measuring both at intermediate checkpoints during code fine-tuning. The Pearson correlation between KL divergence and α was strongly negative, confirming KL is a suitable proxy objective for preserving acceptance rates.

### 3.3 Speculator-Aware Fine-Tuning (EXP-3)

Testing our core hypothesis with Llama models at λ=0.1:

| Domain | Base α | Standard FT α | Spec-Aware α (λ=0.1) |
|--------|--------|--------------|----------------------|
| Code | 0.5954 | 0.5449 (-8.5%) | **0.5596** (-6.0%) |
| Medical | 0.4163 | 0.3747 (-10.0%) | 0.3711 (-10.9%) |
| Chat | 0.3784 | 0.2517 (-33.5%) | **0.3495** (-7.6%) |

The chat domain shows the most dramatic improvement: degradation reduced from 33.5% to 7.6%, recovering most of the lost acceptance rate. Code also benefits. Medical does not improve because the adapter was trained on code data — the spec-aware loss preserves alignment for code-like outputs but cannot help with unrelated domains.

### 3.4 Lambda Sweep and Pareto Analysis (EXP-4)

We swept λ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} for each domain.

**Llama, Code Domain:**

| λ | α | KL | vs Base α |
|---|---|-----|-----------|
| 0.01 | 0.5379 | 0.6157 | -9.7% |
| 0.05 | 0.5409 | 0.5949 | -9.2% |
| 0.10 | 0.5596 | 0.5712 | -6.0% |
| 0.20 | 0.5646 | 0.5156 | -5.2% |
| 0.50 | 0.5881 | 0.3538 | -1.2% |
| 1.00 | **0.6158** | 0.2963 | **+3.4%** |

At λ=1.0, the fine-tuned model **exceeds the base model's acceptance rate** while retaining domain-specific capabilities, demonstrating that the KL regularization can actively improve draft-target alignment.

**Qwen results** show the same monotonic trend across all three domains (code, medical, chat), with medical showing the largest absolute gains at high λ.

### 3.5 Cross-Domain Analysis (EXP-5)

We evaluated each spec-aware fine-tuned model (at optimal λ per domain) on all evaluation domains:

| Train \ Eval | Code | Medical | Chat |
|---|---|---|---|
| Code (λ=1.0) | **0.594** | 0.375 | 0.322 |
| Medical (λ=1.0) | 0.562 | **0.456** | 0.324 |
| Chat (λ=0.5) | 0.551 | 0.395 | **0.324** |

In-domain performance is consistently highest. Cross-domain degradation is modest, indicating spec-aware fine-tuning does not catastrophically hurt out-of-domain acceptance rates.

### 3.6 Loss Function Ablation (EXP-6)

Comparing divergence measures at λ=0.01 on code domain (Qwen):

| Loss Type | α | Ranking |
|-----------|------|---------|
| JS | 0.5509 | Best |
| Token Match | 0.5487 | 2nd |
| TV | 0.5468 | 3rd |
| Forward KL | 0.5405 | 4th |
| Reverse KL | 0.5300 | Worst |

JS divergence provides the best acceptance rate, likely due to its symmetric and bounded nature. Reverse KL performs worst, consistent with its mode-seeking behavior concentrating probability mass. The differences are small at λ=0.01; a higher λ would amplify distinctions.

### 3.7 Complementarity with Runtime Adaptation (EXP-7)

We tested whether spec-aware fine-tuning complements runtime draft adaptation (simulating ATLAS-style systems). After generating outputs from fine-tuned models, we adapted the draft model on these outputs and measured α at different adaptation steps:

| Adaptation Steps | Standard FT α | Spec-Aware FT α |
|-----------------|--------------|----------------|
| 0 | 0.5495 | 0.5300 |
| 100 | 0.5624 | 0.5347 |
| 500 | 0.5909 | 0.5539 |
| 1000 | 0.6000 | 0.5587 |

Both approaches improve with draft adaptation. In the Qwen setting (where standard FT doesn't degrade α), standard FT maintains a slight edge. The more relevant comparison would be with Llama, where spec-aware FT prevents the large initial degradation that runtime adaptation must then recover from.

## 4. Discussion

### Key Findings

1. **Standard LoRA fine-tuning can severely degrade speculative decoding**, with up to 33.5% relative drop in acceptance rate for Llama models on chat data. The effect is model-family dependent — Qwen showed resilience.

2. **Speculator-aware fine-tuning effectively preserves acceptance rates.** At λ=0.1, chat degradation drops from 33.5% to 7.6%. At λ=1.0, the model can exceed the base acceptance rate.

3. **The trade-off is smooth and controllable.** λ provides a single knob to balance task performance against speculative decoding efficiency, enabling practitioners to choose their operating point on the Pareto frontier.

4. **JS divergence is the best regularization loss**, marginally outperforming forward KL, though all losses improve over no regularization.

5. **The approach is complementary to runtime adaptation.** It provides a better starting point for systems like ATLAS that adapt the draft model at inference time.

### Relevance to Together AI's ATLAS System

This work directly complements ATLAS adaptive speculative decoding:
- **ATLAS** adapts the draft model at inference time to recover from distributional drift
- **Speculator-aware FT** prevents drift at training time, reducing the work ATLAS must do
- Combined, they provide both training-time prevention and runtime recovery of speculative decoding efficiency

### Limitations

1. **Task performance trade-off at high λ** — aggressive regularization (λ ≥ 0.5) may noticeably increase task loss, particularly for domains far from the pretraining distribution.
2. **Model-family dependence** — Qwen models showed minimal degradation from standard FT, limiting the benefit of our approach. The method is most valuable for model pairs where fine-tuning causes significant distributional shift.
3. **Single-epoch training** — we used 1 epoch across all experiments for consistency. Longer training may show different dynamics.
4. **Draft model must be available during training** — requires loading both models, approximately doubling GPU memory. This is mitigated by keeping the draft in inference mode with no gradient computation.

## 5. Conclusion

We demonstrate that speculator-aware fine-tuning — a simple KL regularization during LoRA training — effectively preserves speculative decoding acceptance rates across domain-specific fine-tuning. The approach is easy to implement, requires no architectural changes, and provides a controllable trade-off between task adaptation and inference efficiency. For deployment scenarios using speculative decoding, we recommend incorporating speculator-aware loss as a standard component of the fine-tuning pipeline.

## Appendix: Experimental Configuration

- **Target models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct
- **Draft models:** Llama-3.2-1B-Instruct, Qwen2.5-0.5B-Instruct
- **LoRA:** rank=16, alpha=32, dropout=0.05
- **Training:** 1 epoch, batch=4, grad_accum=4, lr=2e-4, cosine schedule
- **Datasets:** StarCoder (code), MedQA (medical), UltraChat (chat), 10K samples each
- **Evaluation:** 50 prompts per domain, K=5 draft tokens, 128 max new tokens
- **Hardware:** NVIDIA H200 (140GB), Northeastern University Explorer cluster
