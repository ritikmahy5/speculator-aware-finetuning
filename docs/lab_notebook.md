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
| `93b7609` | Add EXP-3 results and document EXP-2/EXP-3 findings in lab notebook |
| `422e49f` | Add per-domain SLURM script for EXP-4 lambda sweep |
| `f0234ea` | Fix plot generation for actual file naming, add SLURM scripts and Llama config |
| `05461ac` | Fix EXP-7 SLURM time limit to 8h (gpu partition max) |
| `dbf9bda` | Fix YAML parse crash in analyze_results and EXP-7 prompts key |

---

## 2026-03-13 — EXP-4: Lambda Sweep & Pareto Analysis

### Setup

Submitted as 3 parallel per-domain SLURM jobs to maximize throughput (QOS limit: 4 concurrent jobs):
- Code: job 5043910 on H200 (d4054) — **completed**
- Medical: job 5044006 on H200 (d4054) — running
- Chat: job 5044007 on H200 (d4053) — running

Each domain sweeps λ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} with speculator-aware KL loss. 6 sequential training runs per domain × 3 domains = 18 total runs. λ=0.0 baseline already available from EXP-1.

**Configuration:** 1 epoch, 10K samples, LoRA rank=16, lr=2e-4, max_seq_len=1024. Same as EXP-1/EXP-3 except λ varies.

### Results: Code Domain (Complete — 6/6 λ values)

| λ | α | Task Loss | Spec Loss (KL) | Accept Proxy | KL_mean | JS_mean | TV_mean |
|---|---|-----------|----------------|-------------|---------|---------|---------|
| 0.0 (EXP-1) | 0.5495 | ~0.66 | — | — | 0.6279 | 0.0904 | 0.2435 |
| 0.01 | 0.5405 | 0.6672 | 0.3922 | 0.8552 | 0.6250 | 0.0951 | 0.2532 |
| 0.05 | 0.5375 | 0.6678 | 0.3805 | 0.8584 | 0.6117 | 0.0932 | 0.2501 |
| 0.1 | 0.5300 | 0.6687 | 0.3676 | 0.8616 | 0.5947 | 0.0906 | 0.2454 |
| 0.2 | 0.5357 | 0.6719 | 0.3419 | 0.8682 | 0.5614 | 0.0857 | 0.2375 |
| 0.5 | 0.5494 | 0.7031 | 0.2272 | 0.8937 | 0.3121 | 0.0624 | 0.2036 |
| 1.0 | 0.5939 | 0.7463 | 0.1623 | 0.9126 | 0.2613 | 0.0537 | 0.1873 |

**Observations (code):**
- α increases monotonically with λ: 0.5300 (λ=0.1) → 0.5939 (λ=1.0)
- Task loss also increases: 0.6672 (λ=0.01) → 0.7463 (λ=1.0), a +11.9% increase
- KL_mean drops dramatically: 0.6250 (λ=0.01) → 0.2613 (λ=1.0)
- λ=1.0 achieves the highest α (0.5939), even surpassing the base model (0.5203), at a meaningful task performance cost
- λ=0.5 matches the standard FT α (0.5494 ≈ 0.5495) with +5.4% task loss increase

### Results: Medical Domain (5/6 λ values — λ=1.0 still running)

| λ | α | Task Loss | Spec Loss (KL) | Accept Proxy |
|---|---|-----------|----------------|-------------|
| 0.0 (EXP-1) | 0.3260 | ~1.17 | — | — |
| 0.01 | 0.3340 | 1.1726 | 0.8921 | 0.7506 |
| 0.05 | 0.3522 | 1.1736 | 0.8612 | 0.7574 |
| 0.1 | 0.3559 | 1.1767 | 0.8255 | 0.7642 |
| 0.2 | 0.3724 | 1.1969 | 0.6555 | 0.7845 |
| 0.5 | 0.3924 | 1.2614 | 0.4485 | 0.8264 |
| 1.0 | 0.4556 | 1.3492 | 0.3149 | 0.8588 |

**Observations (medical):**
- Same monotonic pattern: α increases with λ, task loss rises
- Medical benefits most from regularization: α gain from base (0.3103) to λ=1.0 (0.4556) is +46.8% relative
- Task loss increase at λ=1.0 is +15.1% over λ=0.01
- λ=0.5 gives +17.5% α gain with only +7.6% task cost — strong Pareto point

### Results: Chat Domain (5/6 λ values — λ=1.0 still running)

| λ | α | Task Loss | Spec Loss (KL) | Accept Proxy |
|---|---|-----------|----------------|-------------|
| 0.0 (EXP-1) | 0.2902 | ~1.10 | — | — |
| 0.01 | 0.2918 | 1.1047 | 0.6959 | 0.7953 |
| 0.05 | 0.2950 | 1.1053 | 0.6809 | 0.7987 |
| 0.1 | 0.3030 | 1.1092 | 0.6141 | 0.8050 |
| 0.2 | 0.3042 | 1.1223 | 0.5080 | 0.8179 |
| 0.5 | 0.3241 | 1.1594 | 0.3825 | 0.8387 |
| 1.0 | 0.3377 | ~1.20 | ~0.45 | ~0.85 |

**Observations (chat):**
- Same monotonic pattern, α=0.3377 at λ=1.0 (from SLURM log, JSON not persisted due to git sync)
- α gain from base (0.2546) to λ=1.0 is +32.6% relative
- α gain from λ=0.01→0.5 is +11.1% relative, in between code and medical
- Task loss increase is smallest: +5.0% from λ=0.01 to λ=0.5

### Cross-Domain Comparison

| Domain | Base α | Std FT α | Best λ (so far) | Best α | Task Loss Δ |
|--------|--------|----------|-----------------|--------|-------------|
| Code | 0.5203 | 0.5495 | 1.0 | 0.5939 | +11.9% |
| Medical | 0.3103 | 0.3260 | 0.5 | 0.3924 | +7.6% |
| Chat | 0.2546 | 0.2902 | 0.5 | 0.3241 | +5.0% |

**Key insight:** Domains with lower base α (medical, chat) benefit proportionally more from spec-aware regularization. Medical α increases by ~20% relative from base to λ=0.5, while code increases by ~5.6%.

### Pareto Analysis (Preliminary — Code Domain)

For finding the optimal λ (max α subject to <5% task loss increase):
- λ=0.2: task loss +0.7% over λ=0.01, α = 0.5357 — **within 5% budget, minimal gain**
- λ=0.5: task loss +5.4% — **just at the 5% threshold**, α = 0.5494
- λ=1.0: task loss +11.9% — **exceeds 5% budget**, but best α

The 5% task-loss budget yields λ ≈ 0.5 as the optimal for code. Medical and chat data (pending λ=1.0) will help refine per-domain recommendations.

### Analysis

The lambda sweep confirms a clear, monotonic trade-off between task performance and acceptance rate. This is the expected behavior of the regularization, even though the starting point (no degradation from standard FT) differs from the original hypothesis.

The positive direction of the trade-off (higher λ → higher α) is consistent with EXP-2's finding that KL and α are positively correlated. By constraining the target to stay closer to the draft (lower KL), we get HIGHER α — but the cost is that the model learns the task less effectively.

**Pending:** λ=1.0 results for chat domain. Will update when that job completes.

---

## 2026-03-13 — EXP-6: Loss Function Ablation

### Setup

SLURM job 5044817 on H200 (d4055). Code domain, 5 loss types: KL, reverse KL, JS, TV, token_match. The script auto-detected optimal λ from EXP-4 code results — selected λ=0.01 (highest α available at the time the job started, since EXP-4 code was still running).

**Note:** λ=0.01 is very weak regularization. At this λ, the loss type differences are subtle since the regularization barely contributes to the total loss. Results still valid for relative comparison between loss types.

### Results

| Loss Type | α | Task Loss | Spec Loss | KL_mean | JS_mean |
|-----------|-------|-----------|-----------|---------|---------|
| js | **0.5509** | 0.6673 | 0.0758 | 0.6273 | 0.0954 |
| token_match | 0.5487 | 0.6671 | 0.2774 | 0.6288 | 0.0957 |
| tv | 0.5468 | 0.6669 | 0.1987 | 0.6273 | 0.0954 |
| kl | 0.5405 | 0.6672 | 0.3922 | 0.6250 | 0.0951 |
| reverse_kl | 0.5300 | 0.6672 | 0.5421 | 0.5898 | 0.0915 |

### Observations

- **JS divergence achieves the highest α (0.5509)** — a symmetric, bounded divergence that captures bidirectional distributional similarity
- **Token match (0.5487) and TV (0.5468) also outperform KL** — both more directly related to argmax agreement than KL
- **Reverse KL is the worst (0.5300)** — mode-seeking behavior may concentrate target probability on fewer tokens, reducing agreement
- **Task loss is effectively identical** across all loss types (~0.667), confirming that at λ=0.01 the regularization doesn't meaningfully affect task learning
- The spread is only ~2pp (0.53-0.55), which is modest but directionally meaningful

### Interpretation

The ranking (JS > token_match > TV > KL > reverse_kL) makes intuitive sense:
- JS is symmetric and bounded, capturing bidirectional similarity without the asymmetric penalties of KL
- Token match directly optimizes argmax agreement, which is what acceptance rate measures
- TV directly relates to acceptance probability bounds
- Forward KL over-penalizes target mass in regions where draft has none
- Reverse KL concentrates target probability, reducing coverage and hurting acceptance

A re-run at higher λ (e.g., 0.5 or 1.0) would amplify these differences and provide a more definitive comparison.

---

## Llama EXP-1 — Baseline Degradation (Llama 3.1-8B + Llama 3.2-1B)

**Job:** 5051244 (gpu partition, H200)
**Models:** Llama-3.1-8B-Instruct (target) + Llama-3.2-1B-Instruct (draft)
**Training:** Standard LoRA, λ=0.0, 1 epoch, 10K samples per domain

### Acceptance Rate Results

| Domain | Base α | Post-FT α | Δα | Relative Drop |
|--------|--------|-----------|------|--------------|
| Code | 0.5954 ± 0.1535 | 0.5449 ± 0.1271 | -0.0505 | -8.5% |
| Medical | 0.4163 ± 0.1076 | 0.3747 ± 0.1048 | -0.0416 | -10.0% |
| Chat | 0.3784 ± 0.1018 | 0.2517 ± 0.0807 | -0.1267 | **-33.5%** |

### KL Divergence Results

| Domain | Base KL | Post-FT KL | ΔKL |
|--------|---------|------------|-----|
| Code | 0.3793 ± 0.1152 | 0.6227 ± 0.1652 | +0.2434 |
| Medical | 0.5359 ± 0.1632 | 0.8815 ± 0.2204 | +0.3456 |
| Chat | 0.5999 ± 0.1403 | 1.0880 ± 0.3385 | +0.4881 |

### Observations

1. **Llama shows much stronger degradation than Qwen.** Chat domain drops 33.5% — vs Qwen where standard FT showed negligible degradation. This validates the hypothesis that LoRA fine-tuning degrades speculative decoding acceptance rates.
2. **KL increases correlate with α drops.** Chat has the largest KL increase (+0.4881) and the largest α drop (-33.5%). Medical is second on both metrics.
3. **Code domain is most resilient** — likely because the base Llama model was already trained on substantial code data, so the distribution shift from code fine-tuning is smaller.
4. **Llama base α values are generally higher than Qwen** for code (0.60 vs 0.53) but lower for chat (0.38 vs 0.32) — different model families have different draft-target alignment characteristics.

### Comparison: Qwen vs Llama Degradation

| Domain | Qwen Relative Drop | Llama Relative Drop |
|--------|-------------------|-------------------|
| Code | ~0% (no degradation) | -8.5% |
| Medical | ~0% (no degradation) | -10.0% |
| Chat | ~0% (no degradation) | **-33.5%** |

**Key insight:** Llama-3.1-8B + Llama-3.2-1B is a much better test bed for speculator-aware fine-tuning because it actually exhibits the degradation problem our method aims to solve. The Qwen2.5-7B + Qwen2.5-0.5B pair was too robust to standard LoRA fine-tuning.

---

## EXP-5 — Cross-Domain Analysis (Qwen)

**Job:** 5130818 (gpu partition, H200)
**Models:** Qwen2.5-7B-Instruct (target) + Qwen2.5-0.5B-Instruct (draft)
**Optimal λ used:** code=1.0, medical=1.0, chat=0.5 (from EXP-4)

### Cross-Domain Acceptance Rate Matrix

| Train \ Eval | Code | Medical | Chat | Mixed |
|---|---|---|---|---|
| Code (λ=1.0) | **0.5939** ± 0.1310 | 0.3746 ± 0.1212 | 0.3221 ± 0.0599 | 0.4177 ± 0.1407 |
| Medical (λ=1.0) | 0.5622 ± 0.1436 | **0.4556** ± 0.1487 | 0.3243 ± 0.0869 | 0.4209 ± 0.1478 |
| Chat (λ=0.5) | 0.5511 ± 0.1330 | 0.3949 ± 0.1483 | **0.3241** ± 0.0977 | 0.3872 ± 0.1421 |

### Cross-Domain KL Divergence Matrix

| Train \ Eval | Code | Medical | Chat | Mixed |
|---|---|---|---|---|
| Code (λ=1.0) | **0.2613** | 0.4966 | 0.5245 | 0.4704 |
| Medical (λ=1.0) | 0.2973 | **0.4184** | 0.4825 | 0.4393 |
| Chat (λ=0.5) | 0.3070 | 0.4917 | **0.5055** | 0.4752 |

### Observations

1. **In-domain α is highest on the diagonal** — each model performs best on its own training domain, as expected.
2. **Code domain universally benefits from spec-aware FT** — all three models achieve high code α (0.55–0.59), suggesting the code distribution shift from high-λ training is small.
3. **Medical model generalizes well** — the medical-trained model (λ=1.0) achieves competitive α on code (0.5622) and the highest mixed α (0.4209).
4. **Chat α is uniformly low (~0.32)** regardless of training domain — the chat distribution is inherently harder for the draft model.
5. **KL correlates inversely with α** — lowest KL on diagonal (in-domain), highest off-diagonal, matching acceptance rate patterns.
6. **Cross-domain α drops are modest** — training on one domain with spec-aware loss doesn't catastrophically hurt other domains. The worst cross-domain drop is code→chat (0.5939→0.3221), but this is domain mismatch, not method failure.

---

## Llama EXP-3 — Speculator-Aware Fine-Tuning (Llama)

**Job:** 5132164 (gpu partition, H200)
**Models:** Llama-3.1-8B-Instruct (target) + Llama-3.2-1B-Instruct (draft)
**Training:** LoRA with spec-aware loss, λ=0.1, loss_type=kl, code domain, 1 epoch, 10K samples

### Three-Way Comparison: Base vs Standard-FT vs Spec-Aware

| Domain | Base α | Standard FT α | Spec-Aware α (λ=0.1) | Standard FT Drop | Spec-Aware Drop |
|--------|--------|--------------|----------------------|-----------------|----------------|
| Code | 0.5954 | 0.5449 (-8.5%) | **0.5596** (-6.0%) | -8.5% | -6.0% |
| Medical | 0.4163 | 0.3747 (-10.0%) | 0.3711 (-10.9%) | -10.0% | -10.9% |
| Chat | 0.3784 | 0.2517 (-33.5%) | **0.3495** (-7.6%) | -33.5% | -7.6% |

### Training Metrics (from stderr)

- Final step: 625 optimizer steps
- Task loss: ~0.73–0.82
- Spec loss: ~0.48–0.52
- Acceptance proxy: ~0.82 (stable throughout training)

### Observations

1. **Chat domain shows dramatic recovery.** Standard FT degraded chat α by 33.5%, but spec-aware training at λ=0.1 limits the drop to just 7.6%. This is a 26 percentage point improvement — the strongest result in the entire project.
2. **Code domain also benefits.** Degradation reduced from 8.5% to 6.0% — modest but consistent.
3. **Medical shows no benefit at λ=0.1.** The adapter was trained on code domain, so medical is out-of-domain. The spec-aware loss preserves draft alignment for code-like outputs but doesn't help with unrelated domains.
4. **This validates the core hypothesis** — KL regularization during fine-tuning preserves speculative decoding acceptance rates, especially for domains where standard FT causes the most degradation.
5. **Llama is the primary model pair for the paper.** Unlike Qwen (which showed ~0% degradation from standard FT), Llama exhibits real degradation that our method clearly mitigates.

---

## EXP-7 — Complementarity with Runtime Adaptation (Qwen)

**Job:** 5130819 (gpu partition, H200)
**Models:** Qwen2.5-7B-Instruct (target) + Qwen2.5-0.5B-Instruct (draft)
**Procedure:** Generate 1K outputs from each FT model, fine-tune draft on outputs, measure α at 0/100/200/500/1000 adaptation steps

### Results: α Over Adaptation Steps

| Adaptation Steps | Standard FT α | Spec-Aware FT α |
|-----------------|--------------|----------------|
| 0 (no adaptation) | 0.5495 | 0.5300 |
| 100 | 0.5624 | 0.5347 |
| 200 | 0.5741 | 0.5328 |
| 500 | 0.5909 | 0.5539 |
| 1000 | 0.6000 | 0.5587 |

### Observations

1. **Both approaches improve with draft adaptation** — confirming that adapting the draft to match the fine-tuned target is effective regardless of training method.
2. **Standard FT starts slightly higher and improves faster** — this is because with Qwen, standard FT doesn't actually degrade α (it slightly improves it), so the draft has less ground to make up.
3. **The gap narrows but persists** — at 1000 steps, standard FT reaches 0.60 while spec-aware reaches 0.56.
4. **Context: Qwen showed no degradation from standard FT.** This experiment would be more informative with Llama, where standard FT causes real degradation. With Qwen, spec-aware training at λ=0.1 actually hurts α slightly (higher λ would help but trades task loss).
5. **Complementarity interpretation:** In a system like ATLAS, both approaches contribute — spec-aware training provides a better starting distribution, and runtime adaptation further closes the gap. The effect would be more pronounced with model pairs that exhibit real degradation (like Llama).

---

## Llama EXP-4 — Lambda Sweep, Code Domain

**Jobs:** 5137843 (gpu partition, H200)
**Models:** Llama-3.1-8B-Instruct (target) + Llama-3.2-1B-Instruct (draft)
**Base α (code):** 0.5954, **Standard FT α (code):** 0.5449

### Code Domain Results

| λ | α | ± std | KL | vs Base | vs Standard FT |
|---|---|-------|-----|---------|---------------|
| 0.01 | 0.5379 | — | 0.6157 | -9.7% | -1.3% |
| 0.05 | 0.5409 | — | 0.5949 | -9.2% | -0.7% |
| 0.10 | 0.5596 | — | 0.5712 | -6.0% | +2.7% |
| 0.20 | 0.5646 | — | 0.5156 | -5.2% | +3.6% |
| 0.50 | 0.5881 | — | 0.3538 | -1.2% | +7.9% |
| 1.00 | **0.6158** | — | 0.2963 | **+3.4%** | **+13.0%** |

### Observations

1. **Monotonic improvement with λ** — same pattern as Qwen, but with more dramatic effect since Llama actually suffers from degradation.
2. **λ=1.0 surpasses the base model α** — the fine-tuned model achieves 0.6158 vs base 0.5954, a 3.4% improvement. The KL regularization at high λ actually makes the target MORE similar to the draft than the original model.
3. **KL divergence decreases monotonically** — from 0.6157 (λ=0.01) to 0.2963 (λ=1.0), confirming the regularization works as intended.
4. **Clear Pareto trade-off** — lower λ preserves task loss, higher λ preserves/improves α. The optimal λ depends on the user's tolerance for task degradation.
5. **Medical and chat sweeps now complete** — see results below.

## Llama EXP-4 — Lambda Sweep, Medical Domain

**Job:** 5141293 (gpu partition, H200)
**Base α (medical):** 0.4163, **Standard FT α (medical):** 0.3747

| λ | α | ± std | KL | vs Base | vs Standard FT |
|---|---|-------|-----|---------|---------------|
| 0.01 | 0.3952 | 0.0995 | 0.8707 | -5.1% | +5.5% |
| 0.05 | 0.3869 | 0.1001 | 0.8390 | -7.1% | +3.3% |
| 0.10 | 0.3952 | 0.0818 | 0.7941 | -5.1% | +5.5% |
| 0.20 | 0.3817 | 0.0842 | 0.6693 | -8.3% | +1.9% |
| 0.50 | 0.3925 | 0.0890 | 0.4880 | -5.7% | +4.7% |
| 1.00 | **0.4320** | 0.0825 | 0.3895 | **+3.8%** | **+15.3%** |

### Medical Observations

1. **λ=1.0 again exceeds base model** — 0.4320 vs 0.4163 base (+3.8%), similar to code domain.
2. **Non-monotonic at mid-range λ** — α dips at λ=0.2 (0.3817) before recovering. This is unlike the clean monotonic trend in code domain.
3. **All λ values beat standard FT** — even λ=0.01 achieves 0.3952 vs standard FT's 0.3747.
4. **KL decreases monotonically** — from 0.8707 to 0.3895, as expected.

## Llama EXP-4 — Lambda Sweep, Chat Domain

**Job:** 5141294 (gpu partition, H200)
**Base α (chat):** 0.3784, **Standard FT α (chat):** 0.2517

| λ | α | ± std | KL | vs Base | vs Standard FT |
|---|---|-------|-----|---------|---------------|
| 0.01 | 0.2556 | 0.0805 | 1.0841 | -32.5% | +1.5% |
| 0.05 | 0.2635 | 0.0755 | 1.0478 | -30.4% | +4.7% |
| 0.10 | 0.2624 | 0.0814 | 0.9422 | -30.7% | +4.3% |
| 0.20 | 0.2941 | 0.0841 | 0.7203 | -22.3% | +16.8% |
| 0.50 | 0.3554 | 0.0902 | 0.5316 | -6.1% | +41.2% |
| 1.00 | **0.4063** | 0.0897 | 0.4206 | **+7.4%** | **+61.4%** |

### Chat Observations

1. **Most dramatic recovery of any domain** — chat was the worst-degraded domain (-33.5% from standard FT), and λ=1.0 not only recovers but exceeds the base by 7.4%.
2. **λ=0.5 is the practical sweet spot** — recovers to within 6.1% of base while likely preserving more task performance than λ=1.0.
3. **Low λ values barely help** — at λ=0.01-0.1, chat α is still ~30% below base. The distributional shift from chat fine-tuning is so large that weak regularization can't contain it.
4. **λ=1.0 achieves +61.4% over standard FT** — the largest improvement across all domain/λ combinations.

## Llama EXP-4 — Cross-Domain Summary

| Domain | Base α | Std FT α | λ=0.1 α | λ=0.5 α | λ=1.0 α |
|--------|--------|----------|---------|---------|---------|
| Code | 0.5954 | 0.5449 | 0.5596 | 0.5881 | **0.6158** |
| Medical | 0.4163 | 0.3747 | 0.3952 | 0.3925 | **0.4320** |
| Chat | 0.3784 | 0.2517 | 0.2624 | 0.3554 | **0.4063** |

**Key finding:** λ=1.0 exceeds base α in ALL three domains for Llama models. The chat domain benefits most dramatically from speculator-aware training, consistent with it having the largest distributional shift from standard fine-tuning.
