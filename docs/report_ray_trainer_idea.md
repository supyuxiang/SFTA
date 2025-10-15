## Ray PPO Trainer – Idea Summary and Experiment Plan (Detailed)

### 1) Context and Goal
We extend the PPO training pipeline (Ray-based single-controller) to incorporate a reference-model quality score that evaluates actor outputs and blends this signal into token-level rewards in a controllable way, while ensuring stability and efficiency on vLLM backends.

### 2) Key Design (ray_trainer.py)
- Entropy-aware blending of reference quality scores
  - Compute batch-average response entropy (masked over response tokens).
  - Map entropy to a blending weight w ∈ [0, max_weight] via linear decay: w = max_weight × (1 − avg_entropy / entropy_high_threshold).
  - Higher actor entropy → lower reliance on reference score (trust the model more when it is uncertain is reduced). Logged as `actor/ref_quality/weight` and `actor/ref_quality/avg_entropy`.

- Reference scoring path (fsdp_workers.py)
  - Build dedicated scoring prompts; enforce strict output with a single \box{x}.
  - Few-shot examples provided; a compact fallback template used if tokenized length risks overflow.
  - Post-encoding hard truncation: clip `input_ids/attention_mask` to a conservative allowed length (min(model/tokenizer caps, fallback 768) − safety margin), then recompute `position_ids`.
  - Reduce generation length for scoring (`max_new_tokens` → 64) to avoid vLLM `max_model_len` violations.

- vLLM configuration hooks
  - Support `actor_rollout_ref.rollout.max_model_len` in config and pass via CLI to raise backend limit where possible.
  - Training-side lengths reduced (prompt/response) to control memory and ensure total decoder prompt length ≤ max model len.

### 2.1) End-to-End Data/Compute Flow
1. Driver (RayPPOTrainer.fit): builds a batch (prompts, masks), replicates by `rollout.n`.
2. Rollout: actor (vLLM/HF) generates responses; we create `response_mask` from the trailing length.
3. Reward path:
   - Rule-based reward_fn → token-level score matrix.
   - Optional: reference scoring → sample-level quality score in [0,1]; expand to token level with `response_mask`.
   - KL penalty (if enabled) adjusts token-level rewards.
   - Optional: representation alignment bonus (entropy-triggered).
4. Advantage path: GAE/GRPO/other estimator → advantages/returns.
5. Update: critic (if enabled), then actor; log metrics; checkpoint per schedule.

### 2.2) Equations/Definitions
- Batch-average entropy over response region:
  - For entropy matrix E ∈ ℝ^{B×T}, response mask M ∈ {0,1}^{B×T}:
    avg_entropy = mean_b( sum_t(E_b,t × M_b,t) / max(1, sum_t M_b,t) ).
- Linear entropy→weight mapping:
  - Let ê = clip(avg_entropy / τ, 0, 1), where τ is `entropy_high_threshold`.
  - w = (1 − ê) × w_max, clipped to [0, w_max].
- Token-level blending of rewards R and reference scores Q (both masked):
  - R' = (1 − w) × R + w × Q.
- KL penalty (example): R'' = R' − β × KL, masked over response tokens.

### 2.3) Implementation Pointers
- Entropy to weight helpers: `entropy_to_weight_linear/exp/cosine/piecewise` (ray_trainer.py).
- Reference scoring entry: `_compute_ref_quality_score` (ray_trainer.py → fsdp_workers.generate_quality_score)。
- Safety on vLLM: compact template; post-encoding clipping; `max_new_tokens=64`; optional `max_model_len` in config.


### 3) Why This Works
- Entropy-aware blending:
  - When actor is highly certain (low entropy), keep a higher reference influence to stabilize reward shaping.
  - When actor is uncertain (high entropy), rely less on a possibly noisy reference score.
  - Smooth linear policy is simple, interpretable, and easy to tune via two hyperparameters.

- Robust scoring on vLLM:
  - Compact prompt + token-level clipping + shorter `max_new_tokens` prevents overlength errors.
  - Recomputed `position_ids` ensures downstream kernels receive consistent inputs after clipping.

### 4) Current Default/Important Params
- Algorithm.ref_quality_score:
  - `enable: True`, `max_weight: 0.3`, `entropy_high_threshold: 1.0` (used by entropy→weight mapping)
- Scoring generation: `max_new_tokens: 64` (fsdp_workers)
- Length control (script): `data.max_prompt_length: 256`, `data.max_response_length: 256`
- vLLM: `actor_rollout_ref.rollout.max_model_len: 2048` (configurable)

### 5) Experiment Plan (Ablations and Scaling)
- A. Blending strategy ablation
  - A1 Linear (current)
  - A2 Cosine decay (entropy_to_weight_cosine)
  - A3 Exponential decay (entropy_to_weight_exp)
  - A4 Piecewise (full/linear/zero)
  - Metrics: val reward/acc on GSM8K-like sets; training stability (loss spikes); final PPO returns

- B. Max weight sensitivity
  - `max_weight ∈ {0.1, 0.2, 0.3, 0.4}` (fix threshold=1.0)
  - Identify the knee where added reference help plateaus or destabilizes

- C. Entropy threshold sensitivity
  - `entropy_high_threshold ∈ {0.6, 0.8, 1.0, 1.2}` (fix max_weight=0.3)
  - Find the best pivot for decay matching current model entropy range

- D. Prompt/scoring robustness
  - D1 With few-shot vs compact template only
  - D2 `max_new_tokens ∈ {32, 64, 96}` for scoring
  - D3 With/without token-level hard clipping
  - Monitor vLLM overlength errors and scoring coverage (non-zero scores ratio)

- E. Compute efficiency
  - Compare GPU memory, step time and throughput under length settings {256/256, 320/320} and max_model_len {1024, 2048}

- F. Device placement for scoring (optional)
  - Pin reference scoring worker to a spare GPU (e.g., cuda:9) via a dedicated Ray resource pool; measure overlap impact.

### 6) Reporting Metrics and Logging
- Core
  - `val-core/*/reward/mean@1` (or acc where available)
  - `actor/ref_quality/weight`, `actor/ref_quality/avg_entropy`
  - Non-zero score rate from reference scoring; score distribution (mean/std)
- Efficiency/Health
  - `perf/*` (throughput, memory), step time, OOM/overlength incident counts

Suggested plots/tables:
- Weight vs avg entropy over time; correlation with validation metric.
- Non-zero reference score ratio vs configuration.
- Step time & memory vs max_model_len & (prompt_len,response_len).

### 7) Risks and Mitigations
- Risk: Overlength prompts causing reference scoring to fail
  - Mitigation: compact template, token-level clipping, lower `max_new_tokens`, increase backend `max_model_len` when possible
- Risk: Over-regularization from reference blending
  - Mitigation: entropy-aware decay; tune `max_weight` and threshold; run A/B on baselines

- Risk: Score extraction failures (no \box{...} or decoding anomalies)
  - Mitigation: token-level parser fallback, strict end-only box format; default 0.5 only if both paths fail.

- Risk: DRIFT from reward function (distribution shift)
  - Mitigation: schedule w_max smaller at start; periodic sanity checks using known GSM8K items.

### 8) Next Steps Checklist
- [ ] Land entropy-based blending (done) and keep the old schedule code commented for fallback (done)
- [ ] Enable compact scoring template and token clipping globally (done)
- [ ] Run A/B on linear vs cosine vs exp vs piecewise (grid above)
- [ ] Sweep max_weight and entropy_high_threshold
- [ ] Tune `data.max_prompt_length/response_length` and `max_model_len` to a stable pair
- [ ] Track non-zero scoring ratio and fix any residual failure modes

Deliverables per milestone:
- M1 (stability): runbooks + logs; no overlength; linear blending base.
- M2 (ablations): summary tables (A/B), best config per metric; plots.
- M3 (write-up): final recommended defaults; PR with guards and docs.

### 9) Tentative Timeline
- Week 1: Stabilization runs (lengths, clipping, `max_model_len`); linear vs cosine
- Week 2: Threshold/weight sweeps, compact vs few-shot scoring
- Week 3: Consolidated results, write-up and plots; choose default

### 10) Resource & Carbon Estimate (rough)
- Hardware: 4×A800(80G) or 4×A100(80G) equivalent; single-node Ray.
- Per run (256/256 lengths, vLLM, linear blending): ~0.5–1.5 GPU-hours; ablation grid (≈12–20 runs) → 6–30 GPU-hours.
- Carbon note: Prefer night-time low-carbon slots; cache models on NVMe; set `gpu_memory_utilization ≤ 0.65` to reduce peaks.


