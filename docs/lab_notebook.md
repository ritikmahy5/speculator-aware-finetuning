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

### What's next (Day 2)
- Run smoke test on Explorer cluster to verify end-to-end pipeline
- Begin EXP-1: measure base acceptance rate, then fine-tune on code domain
- Start collecting real experimental data
