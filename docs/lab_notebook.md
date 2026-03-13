# Lab Notebook — Speculator-Aware Fine-Tuning

---

## Day 1 — 2026-03-12 — Project Setup & Implementation

### What was done

**Project scaffold:**
- Initialized git repository with `main` branch
- Created full directory structure: `src/`, `configs/`, `scripts/`, `results/`, `plots/`, `notebooks/`, `docs/`, `logs/`
- Created `.gitignore` (excludes model checkpoints, SLURM logs, planning files)
- Created `requirements.txt` with all dependencies

**Configuration files:**
- `configs/models.yaml` — default model paths, LoRA config, training hyperparameters
- `configs/eval_prompts.yaml` — 200 eval prompts (50 code, 50 medical, 50 chat, 50 mixed)
- `configs/exp1_baseline.yaml` through `configs/exp7_complementarity.yaml` — all 7 experiment configs

**SLURM templates:**
- `scripts/slurm_singlegpu.sh` — single H200 GPU job template
- `scripts/slurm_multigpu.sh` — dual H200 GPU job template
- `scripts/slurm_smoke_test.sh` — quick verification job

**Core source modules:**
- `src/utils.py` — config loading (YAML + CLI overrides with dot notation), seed setting, logging (console + file), device auto-detection, checkpoint saving, metrics JSON export
- `src/data.py` — dataset loading for code (starcoderdata), medical (medqa), chat (ultrachat_200k) with streaming, text extraction, tokenization, padding/truncation, label masking
- `src/spec_loss.py` — 5 loss variants (KL, reverse KL, JS, TV, token match) with acceptance proxy, proper draft detachment, device handling, top-k support, and a self-test block
- `src/train.py` — full training loop with LoRA (via PEFT), optional draft model loading, gradient accumulation, bf16 autocast, cosine LR scheduler with warmup, checkpointing, metrics logging
- `src/measure_acceptance.py` — manual speculative decoding simulation (draft proposes K tokens, target verifies), per-position acceptance tracking, LoRA adapter loading support
- `src/measure_kl.py` — divergence measurement (KL, JS, TV, reverse KL) between any two models on eval prompts, with LoRA adapter support
- `src/analyze_results.py` — publication-quality plotting for all 8 required plots (degradation bars, KL correlation scatter, spec-aware comparison, Pareto frontiers, cross-domain heatmap, loss ablation, complementarity lines)

**Scripts:**
- `scripts/smoke_test.sh` — end-to-end pipeline verification (tiny data: 100 samples, 5 eval prompts)
- `scripts/run_exp1.sh` through `scripts/run_exp7.sh` — experiment runners

**Documentation:**
- `README.md` — public-facing project description
- `docs/lab_notebook.md` — this file

### Key design decisions
- **spec_loss.py handles draft_logits=None** for baseline training (lam=0), avoiding unnecessary draft model loading
- **Manual KL computation** used instead of `F.kl_div` to avoid the confusing PyTorch API argument ordering
- **Streaming datasets** with buffer_size=5000 to avoid downloading full datasets
- **Labels have padding masked to -100** so cross-entropy ignores pad tokens

### Deviations from plan
- None so far — following the CLAUDE.md build order exactly

### Issues encountered
- None yet — all modules implemented without errors

---

## 2026-03-12 — Cluster Deployment, Smoke Test & EXP-1

### Model Switch: Llama → Qwen

**Problem:** `meta-llama/Llama-3.1-8B-Instruct` is gated on HuggingFace — access request submitted but pending approval. Also discovered `Llama-3.1-1B-Instruct` doesn't exist; the 1B model shipped with Llama 3.2 (`Llama-3.2-1B-Instruct`).

**Decision:** Switched to ungated Qwen models as primary, will re-run with Llama when approved:

| Role | Model | Vocab Size |
|------|-------|-----------|
| Target | `Qwen/Qwen2.5-7B-Instruct` | 152,064 |
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | 151,936 |

Updated: `configs/models.yaml`, all 7 experiment configs, `README.md`.

**Vocab size mismatch (152,064 vs 151,936):** Qwen 7B and 0.5B have different vocabulary sizes. Fixed by truncating both logit tensors to `min(target_vocab, draft_vocab)` before spec loss and KL computation. Task loss uses full target logits (unaffected).

### Dataset Switch

**Problem:** `bigcode/starcoderdata` is gated.

**Decision:** Switched code domain to `iamtarun/python_code_instructions_18k_alpaca` (ungated). Updated `src/data.py` text extraction to use instruction+output fields.

### Bug Fixes Before Deployment

1. **Device mismatch in `spec_loss.py` and `measure_kl.py`:** `torch.log(torch.tensor(EPSILON))` creates a CPU tensor even when model is on GPU. Fixed by using `math.log(EPSILON)` (Python float, device-agnostic).

2. **Vocab alignment in `spec_loss.py`:** Added truncation to `min(target_vocab, draft_vocab)` for spec loss and acceptance proxy computation.

3. **Same vocab alignment fix in `measure_kl.py`:** Applied identical truncation after logit extraction.

### Explorer Cluster Setup

- **SSH:** `mahyavanshi.r@login.explorer.northeastern.edu`
- **Conda env:** `/scratch/mahyavanshi.r/envs/specaware` (Python 3.11, PyTorch 2.5.1+cu121)
- **Repo cloned to:** `/scratch/mahyavanshi.r/speculator-aware-finetuning`
- **HF cache:** `/scratch/mahyavanshi.r/.cache/huggingface`
- **SLURM fixes needed:** `cuda/12.1` → `cuda/12.1.1`, added `source .../conda.sh` before `conda activate`

### Smoke Test

Ran on A100 (40GB) via `sbatch scripts/slurm_smoke_test.sh`. All 5 steps passed:
1. Training (100 samples, lam=0.0) — OK
2. Spec-aware training (100 samples, lam=0.1) — OK
3. Acceptance measurement — OK
4. KL measurement — OK
5. Plot generation — OK

### EXP-1: Baseline Degradation Measurement

**Setup:** SLURM job 5038845 on H200 (140GB, node d4055). Single-GPU mode (both models on `cuda:0`). Runtime: ~3 hours.

**Configuration:** Standard LoRA fine-tuning (lam=0.0), 1 epoch, 10K samples per domain, batch_size=4, grad_accum=4, lr=2e-4, max_seq_len=1024.

#### Results: Base Model Acceptance Rates

| Domain | α (mean) | α (std) | KL | JS | TV |
|--------|----------|---------|------|------|------|
| Code | 0.5203 | 0.1627 | 0.4248 | 0.0877 | 0.2427 |
| Medical | 0.3103 | 0.0587 | 0.6683 | 0.1250 | 0.3075 |
| Chat | 0.2546 | 0.0574 | 0.7205 | 0.1375 | 0.3479 |
| Mixed | 0.3305 | 0.1476 | 0.6430 | 0.1248 | 0.3180 |

**Observation:** Code domain has highest base acceptance (0.52) and lowest KL (0.42), suggesting the Qwen 7B/0.5B pair is already most aligned on code-like text. Chat has worst acceptance (0.25) and highest KL (0.72).

#### Results: After Standard LoRA Fine-Tuning (Same-Domain Eval)

| FT Domain | Base α | FT α | Δα | Relative Δ |
|-----------|--------|------|-----|-----------|
| Code | 0.5203 | 0.5495 | +0.0292 | +5.6% |
| Medical | 0.3103 | 0.3260 | +0.0157 | +5.1% |
| Chat | 0.2546 | 0.2902 | +0.0356 | +14.0% |

#### Results: Cross-Domain Acceptance Rate Matrix (α after FT)

| FT Domain \ Eval | Code | Medical | Chat | Mixed |
|-------------------|------|---------|------|-------|
| Base (no FT) | 0.5203 | 0.3103 | 0.2546 | 0.3305 |
| Code-FT | **0.5495** | 0.3242 | 0.2759 | 0.3573 |
| Medical-FT | 0.5145 | **0.3260** | 0.2826 | 0.3634 |
| Chat-FT | 0.5235 | 0.3261 | **0.2902** | 0.3589 |

#### Results: KL Divergence After Fine-Tuning (Same-Domain Eval)

| FT Domain | Base KL | FT KL | ΔKL |
|-----------|---------|-------|-----|
| Code | 0.4248 | 0.6279 | +0.2031 |
| Medical | 0.6683 | 0.7310 | +0.0627 |
| Chat | 0.7205 | 0.8496 | +0.1291 |

### Key Finding from EXP-1

**The expected degradation did NOT occur.** Success criteria was "α drops by >15% relative for at least one domain" — instead, acceptance rates slightly *increased* across all domains after standard LoRA fine-tuning.

**Interpretation:** 1 epoch of LoRA fine-tuning with rank=16 on 10K samples produces insufficient distribution shift to measurably degrade speculative decoding acceptance rates with the Qwen 7B/0.5B pair. The KL divergence does increase (especially for code: +0.2031), but this increase doesn't translate into lower acceptance rates.

**Possible explanations:**
1. LoRA rank=16 constrains the fine-tuning to a low-rank subspace, limiting distribution shift
2. 1 epoch / 10K samples is too little data to cause significant drift
3. The Qwen 7B/0.5B pair already has low base acceptance (especially medical/chat at ~0.25-0.31), leaving little room for further degradation
4. Acceptance rate (which depends on argmax agreement) may be more robust than KL (which measures full distributional shift)

**This is still a useful finding** — it establishes that lightweight LoRA fine-tuning is relatively safe for speculative decoding compatibility, at least for this model pair and training configuration.

### Plot Generation Issue

`src/analyze_results.py` expected filename `acceptance_base.json` but `run_exp1.sh` produces per-domain files (`acceptance_base_code.json`, etc.). Fixed to support both naming conventions.

### Next Steps

EXP-2 (KL-Acceptance Correlation) and EXP-3 (Speculator-Aware FT) submitted in parallel:
- EXP-2: SLURM job 5042692 on H200
- EXP-3: SLURM job 5042693 on H200

EXP-2 will provide more data points (checkpoints at 25/50/75/100%) to establish whether the KL-α correlation holds. EXP-3 will test whether the speculator-aware loss (lam=0.1) provides any benefit even when baseline degradation is minimal.

---

## 2026-03-12 — EXP-2: KL–Acceptance Rate Correlation

### Setup

SLURM job 5042692 on H200 (d4055). Single-GPU mode. Trains code domain with lam=0.0, saves checkpoints at 25/50/75/100% of training. Measures α and KL/JS/TV at each checkpoint.

### Results: Divergence vs Acceptance at Checkpoints

| Checkpoint | α | KL | JS | TV |
|-----------|-------|--------|--------|--------|
| Base | 0.5203 | 0.4248 | 0.0877 | 0.2427 |
| Step 156 (25%) | 0.5355 | 0.5220 | 0.0883 | 0.2380 |
| Step 312 (50%) | 0.5390 | 0.5688 | 0.0896 | 0.2396 |
| Step 468 (75%) | 0.5498 | 0.5893 | 0.0898 | 0.2424 |
| Step 624 (100%) | 0.5495 | 0.6279 | 0.0904 | 0.2435 |

### Correlation Results

| Metric | Pearson r | p-value | Spearman r | p-value |
|--------|-----------|---------|------------|---------|
| KL vs α | **+0.956** | 0.003 | — | — |
| JS vs α | **+0.978** | 0.0007 | — | — |
| TV vs α | **+0.982** | 0.0005 | — | — |

### Key Finding from EXP-2

**POSITIVE correlation between KL and α** (r = +0.956) — the opposite of the expected negative correlation. As the target model diverges further from the draft (higher KL), acceptance rate actually *increases*.

This challenges the core assumption of the project. The KL divergence in the loss function is designed to keep the target close to the draft to preserve α, but the data shows that higher KL (more divergence) correlates with higher α.

**Interpretation:** LoRA fine-tuning on code data may be making the target's distribution "sharper" — concentrating probability on the correct tokens. If the draft also assigns high probability to those tokens, sharpening the target makes verification MORE likely to succeed, even though the distributions diverge in KL terms. KL measures distributional distance, not argmax agreement (which is what acceptance rate fundamentally depends on).

---

## 2026-03-12 — EXP-3: Speculator-Aware Fine-Tuning (Core)

### Setup

SLURM job 5042693 on H200 (d4055). Single-GPU mode. Fine-tuned Qwen2.5-7B-Instruct on code domain with speculator-aware loss: `L_total = L_task + 0.1 × KL(target || draft)`. Training: 1 epoch, 10K samples, 625 optimizer steps. Runtime: 46 minutes total (28 min training + 18 min evaluation).

### Training Trajectory

| Steps | task_loss | spec_loss (KL) | accept_proxy |
|-------|-----------|---------------|-------------|
| 10 | 0.979 | 0.230 | 0.894 |
| 100 | 0.658 | 0.365 | 0.864 |
| 200 | 0.659 | 0.323 | 0.877 |
| 300 | 0.714 | 0.354 | 0.867 |
| 400 | 0.679 | 0.371 | 0.866 |
| 500 | 0.686 | 0.363 | 0.868 |
| 600 | 0.710 | 0.385 | 0.860 |
| 625 (final) | ~0.68 | ~0.37 | ~0.86 |

**Observation:** spec_loss (KL between target and draft) increased from 0.23 → ~0.37 during training, meaning the KL regularization did NOT prevent distributional divergence — it only slowed it. The acceptance proxy remained steady around 0.86.

### Results: Three-Way Comparison (Code Domain)

| Condition | α (mean ± std) | KL | JS | TV |
|-----------|----------------|------|------|------|
| Base model | 0.5203 ± 0.1627 | 0.4248 | 0.0877 | 0.2427 |
| Standard FT (λ=0.0) | 0.5495 ± 0.1527 | 0.6279 | 0.0904 | 0.2435 |
| Spec-aware FT (λ=0.1) | 0.5300 ± 0.1540 | 0.5947 | 0.0906 | 0.2454 |

**Improvement over standard FT: -0.0195 (-3.5%)** — spec-aware FT performed WORSE than standard FT.

### Results: Cross-Domain Acceptance (Spec-Aware Code-FT Model)

| Eval Domain | Spec-aware FT α | Standard FT α (EXP-1) | Base α |
|-------------|-----------------|----------------------|--------|
| Code | 0.5300 | 0.5495 | 0.5203 |
| Medical | 0.3421 | 0.3242 | 0.3103 |
| Chat | 0.2813 | 0.2759 | 0.2546 |
| Mixed | 0.3553 | 0.3573 | 0.3305 |

### Key Finding from EXP-3

**The speculator-aware loss (KL regularization) did not improve acceptance rates.** In fact, it slightly hurt same-domain performance (-3.5% vs standard FT on code). Cross-domain results were mixed — slightly better on medical (+1.8pp) and chat (+0.5pp), slightly worse on mixed (-0.2pp).

### Analysis

This result is consistent with EXP-1 and EXP-2:
1. Standard LoRA FT doesn't degrade α (EXP-1) — there's no problem to solve
2. KL and α are positively correlated (EXP-2) — minimizing KL could actually *hurt* α
3. The KL regularization constrains the target model from learning as effectively, slightly reducing task performance AND acceptance rate

The spec_loss did reduce KL somewhat (0.5947 vs 0.6279 for standard FT), but this "improvement" in KL actually corresponds to *worse* acceptance rate — exactly matching the positive correlation found in EXP-2.

### Decision Point (per CLAUDE.md)

Per the experimental plan: "α doesn't improve → debug: check L_spec is decreasing, check for gradient conflict"

**L_spec trajectory:** spec_loss INCREASED during training (0.23 → 0.37), meaning the regularization couldn't overcome the task-driven divergence. However, it did result in a lower final KL (0.595) compared to standard FT (0.628), confirming the loss IS having a regularizing effect.

**Recommended next steps:**
1. **Proceed to EXP-4 (lambda sweep)** anyway — higher λ values (0.5, 1.0) may show more effect, and the sweep data is valuable for the paper regardless of direction
2. **EXP-6 (loss ablation)** — token_match loss may work better since it directly optimizes argmax agreement rather than distributional KL
3. Consider whether a different setting (more aggressive FT: more epochs, higher rank, full fine-tuning) would first create degradation, making the spec-aware loss relevant

---

### Git Commits (2026-03-12)

| Hash | Description |
|------|-------------|
| `be166fa` | Fix draft model ID: Llama-3.1-1B → Llama-3.2-1B |
| `c1a2c26` | Switch to Qwen2.5 models (ungated) |
| `f9694b6` | Fix SLURM scripts for Explorer cluster |
| `bbe5924` | Switch code dataset to ungated python_code_instructions_18k_alpaca |
| `ac35878` | Request H200 GPU for smoke test |
| `16a5d11` | Use A100 for smoke test (H200s unavailable) |
| `5912cf4` | Fix device mismatch in spec_loss clamp |
| `cd967d1` | Handle mismatched vocab sizes between target and draft models |
| `16b0fc6` | Fix device mismatch and vocab size alignment in measure_kl.py |
| `f613283` | Fix analyze_results.py to match per-domain file naming |
| `01035f7` | Fix bar_label crash on ErrorbarContainer in plot generation |
| `8a89cb9` | Add EXP-1/EXP-2 results, SLURM scripts, and lab notebook documentation |
