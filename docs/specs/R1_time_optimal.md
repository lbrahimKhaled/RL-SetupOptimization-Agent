# R1: time-optimal reward with potential-based ξref guidance + warm start from full1

Status: rl-architect spec, 26 Oct 2026, binding for rl-coder. Ibrahim approved R1–R4 (26 Oct); CMDP/Lagrangian deferred. Old reward stays the default, so full1 stays reproducible. Labels: [F] formulation, [C] code (read/run here), [P] paper, [J] judgement. Saved verbatim-in-substance by the lead from the architect's reply.

## Architect verdict
| Layer | Verdict |
|---|---|
| R1 reward (progress − time + PBRS, constant P, stall → truncation) | APPROVE-WITH-NOTES (F1; Q1) |
| Shaping level, γ_s, truncation handling, P, k | APPROVE as in §1 |
| R3 warm start (actor + critics + targets + α, fresh buffer, frozen actor 100k) | APPROVE-WITH-NOTES |
| R4 eval additions + stop rule | APPROVE |
| Existing pipeline under the new reward | CHANGES-REQUIRED: 9 silent breaks, fixed in §6 |

**F1.** Under R1 the converged driver does not depend on ξref (Ng et al. 1999 Thm 1: PBRS is policy-invariant). This is the "c → 0 kills the driver axis" case in PROJECT_CONTEXT §4; Ibrahim decided it. It is not v5 (§12): no c annealing (c out of the reward from step 0), driver still doesn't see the setup, simulator is OCD. DECISIONS #19 records it.

## 0. Facts
- [C] full1/seed0 at 2.95M: lap 2 = 123.30 s; costs time 123.30, smoothness 10.4, tracking 175.9, cost 309.6; mean |e| 0.39 m, max 1.83 m, min edge margin 0.22 m, 3,083 decisions.
- [C] best.zip (2.6M): lap 2 = 122.84531256503526 s, stochastic mean 124.77 s, α = 0.0434.
- [C] 250 env steps/s training; eval ≈ 25 s per 50k (≈ 11 %).
- [C] Q_min − MC (per step) +4.5…+12 over mean MC ≈ 190 (2–6 %).
- [C] Per decision: progress ≈ 1.748, time 0.04, tracking 0.057, smoothness 0.0034 → R1 raises Q of the same policy by ≈ 0.060/(1−γ) ≈ +7.6 (≈ 4 %).
- [C] Shanghai width 10.1–17.8 m; ξref-to-edge ≤ 15.1 m → on track |e| ≤ 14.1 m.
- [C] |e| at failure in training 0.8–3.3 m → old P ≈ 52–83.
- [C] SB3 2.9.0 `NStepReplayBuffer._get_samples`: Σ_{j<m} γ^j r_{k+j}, m = steps up to and incl. first done/timeout; bootstrap γ^m; target zeroed only if done and not timeout; float32 discounts.
- [C] `SubprocVecEnv` sets `TimeLimit.truncated = truncated and not terminated`, passes `terminal_observation` (used as next_obs).

## 1. Reward R1 (env)
Per base tick, mode `time_pbrs`: r_tick = a·Δs − b·Δt − [failure]·P. Smoothness and tracking are still computed/logged in `step_cost`/`episode_cost`/`laps`, not subtracted. c and W become diagnostic weights (kept 0.1 and (.01,.001,.001)).

Per decision: R_k = Σ_ticks r_tick + F_k, F_k = γ·Φ(s_{k+1}) − Φ(s_k), Φ(s) = −k_s·|e(s)|, e at the decision-boundary state (= obs[9]·5).

### 1.1 Shaping per decision, γ_s = γ = 0.992 exactly
- Learner discounts per decision with `model.gamma = cfg.env.reward.gamma`; ticks summed undiscounted.
- Per-tick shaping with γ^(1/2) does not telescope (non-potential residual).
- Decision level with γ_s = γ: any n-step window m ≤ 3 gives Σ γ^j F_{k+j} = γ^m Φ(s_{k+m}) − Φ(s_k), exact also for windows cut by done/timeout.
- γ_s ≠ γ → residual (γ_s − γ)Φ(s′) = a real non-invariant L1 line penalty. γ_s must be the same object as the learner's γ (test).
- [J] Holds for soft Q: Q′ = Q − Φ(s), state-only shift.

### 1.2 Episode ends
| End | Flag | Φ(s′) in F | Penalty |
|---|---|---|---|
| off_track, nonfinite, wrong_way | terminated | 0 (absorbing; Linesight same) | P (constant) |
| stall (v < 3 m/s > 2 s) | **truncated** (`truncation='stall'`, `termination=None`) | real −k·\|e(s′)\| | none |
| time limit | truncated | real | none |
| eval `lap_complete` (eval only) | terminated | **real** (keeps Q-vs-MC consistent) | none |

Truncation needs the real Φ: SB3 bootstraps γ^m Q(s′); with Φ = 0 every 60 s cut would pay a spurious +γk|e′|.

### 1.3 k_s = 1.0 per m (PLACEHOLDER, [J])
1 m lateral offset = 1 m progress in potential. Drift at 2 m/s lateral → |F| ≈ 0.08/decision vs net ≈ 1.7 (≤ 5 %). Constant offset d → F = (1−γ)k d per decision (0.08 at 10 m), telescopes away in the discounted objective. PBRS ≡ Q-init with Φ (Wiewiora 2003, to verify) → with warm-started critics the guidance effect is ≈ nil; matters for a from-scratch R1 run. `shaping_k = 0` = pure progress − time ablation.

### 1.4 P = 100 constant (PLACEHOLDER, [J])
Unshaped: per-tick reward Δt(a v − b) > 0 above 1 m/s; worst continuation (stand still forever) = −b·0.04/(1−γ) = −5.0 (stall truncation bootstraps, not an escape); a failure decision earns ≤ 90·0.04 = 3.6 − P. Crashing pays only if P < 8.6. Shaped: Q_shaped = Q − Φ(s), ordering identical; transient crash "bonus" ≤ k·14.1 = 14.1; a myopic critic still scores a crash ≤ 3.6 + 14.1 − 100 = −82 vs continuing ≥ ≈ −1.1. 100 ≈ old 52–83 (slightly more risk-averse, freed line runs nearer the edges) ≈ 0.5·V(s).

### 1.5 Undiscounted lap effect (record only)
Σ_k F_k = γΦ_T − Φ_0 + (1−γ)k Σ_{k=1}^{T−1}|e_k| ≈ +9.9 per lap at mean |e| 0.4 → episode return is not a lap score. Lap costs (`laps`) never include shaping or P; setup agent scores eval-lap cost (DECISIONS #13).

## 2. Code changes
### 2.1 `config.py` `RewardConfig` (defaults keep full1)
```python
mode: str = 'ref_tracking'     # 'ref_tracking' = reward (1), full1 | 'time_pbrs' = R1
shaping_k: float = 1.0         # [1/m] Phi(s) = -shaping_k*|e(s)|; time_pbrs only. PLACEHOLDER
penalty_const: float = 100.    # P for terminal failures in time_pbrs. PLACEHOLDER
```
Docstring: in `time_pbrs` c and w are diagnostic; γ is also the shaping discount. `failure_penalty()` unchanged (legacy). Validate mode in set; shaping_k, penalty_const finite ≥ 0; 0 < gamma ≤ 1. Old config.json without the keys loads as legacy (test).

### 2.2 `env.py`
- `TERMINAL_FAILURES = ('off_track', 'nonfinite', 'wrong_way')`; `FAILURES` keeps its stats meaning.
- **Legacy path byte-for-byte unchanged** (reuse base `r`; stall terminal). Do not "simplify" to the costs dict (op order → bit-identity).
- `time_pbrs` in `step()`: stall → `truncated=True, terminated=False, self.done=True, self._trunc_cause='stall'`, no penalty; failure in TERMINAL_FAILURES → `costs['penalty'] = rw.penalty_const`; `r = costs['progress'] - costs['time'] - costs['penalty']` per tick.
- After the tick loop (time_pbrs only):
  ```python
  phi_next = 0. if (terminated and cause in TERMINAL_FAILURES) else -rw.shaping_k * abs(e_end)
  shaping = rw.gamma * phi_next - self._phi;  total += shaping;  self._phi = phi_next
  ```
  `self._phi` set in `reset()` from `_line_errors()[0]` at hand-over; `e_end = _line_errors()[0]` after the loop.
- info: `shaping` (0. legacy), `phi`, `episode_shaping`, `reward_mode`; `truncation = self._trunc_cause` ('time_limit' | 'stall' | None).
- `_time_laps`: `cost` keeps legacy meaning (time + smoothness + tracking) in both modes; add `cost_r1 = time`. Shaping and P never enter `laps`.
- Update module docstring.

### 2.3 `evaluate.py`
`TRACE_KEYS += 'shaping', 'progress_m'`. `lap_metrics` adds `cost_r1`, `share_abs_e_gt_1m`, `share_abs_e_gt_3m`, `shaping_sum`, `progress_jumps` (decisions with Δprogress > 2·speed·Δt + 1 m or < −1 m). Keeps mean/max |e|, min_edge_margin_m, cost, tracking, smoothness. `evaluate_model` adds `reward_mode`. Q-vs-MC unchanged.

### 2.4 `train_config.py`
```python
@dataclass(frozen=True)
class WarmStartConfig:
    critic_warmup_steps: int = 100_000   # actor + alpha frozen, critics learn R1 online
    load_critics: bool = True            # critics + target critics from init_from
    load_alpha: bool = True              # log_ent_coef from init_from (optimizer fresh)
    min_buffer_batches: int = 10         # no grad step until buffer >= batch_size * this
    critic_check_tol: float = .08        # |Q_min - MC| / |MC| at the unfreeze eval; above -> warning

@dataclass(frozen=True)
class StopConfig:
    plateau: bool = False
    min_steps: int = 1_000_000
    window: int = 500_000
    min_improvement_s: float = .1
```
`TrainConfig += warm, stop, init_from: str | None = None, strict_provenance: bool = True`. `AgentConfig.init_mode` gains `'warm'`. Presets: `full` unchanged; `smoke` strict_provenance=False; `r1_warm`: reward.mode='time_pbrs', init_mode='warm', total_timesteps=3_000_000, max_wall_h=4.0, guard.threshold=0.75, stop.plateau=True; `r1_warm_smoke`: r1_warm + max_wall_h=0.25, critic_warmup_steps=10_000, eval.interval=25_000, eval.stochastic_laps=3, guard.interval=5_000, checkpoint_interval=25_000, stop.plateau=False, strict_provenance=False.

### 2.5 `agent.py` `warm_start(model, init_from, cfg) -> dict`
1. `src = StagedSAC.load(init_from, device='cpu')`.
2. Require equal (else `InitMismatch`): obs space, action space, gamma, policy class, net_arch, n_critics, critic LayerNorm. Read source `config.json` at `Path(init_from).parents[1]`; its `env` dict must equal the new one with `env.reward` removed from both, and `reward.gamma` equal. Missing config.json → refuse.
3. `model.policy.load_state_dict(src.policy.state_dict())` (actor, critic, critic_target). `load_critics=False` → only `actor.*` (strict=False).
4. `load_alpha`: `with th.no_grad(): model.log_ent_coef.copy_(src.log_ent_coef)` (in place).
5. Optimizers stay fresh.
6. No `init_output_bias`. `model.min_train_transitions = batch_size * min_buffer_batches`; `model.set_frozen(True, True)`.
7. Return `{'path','sha256','src_num_timesteps','src_alpha','src_config_sha256'}`.

### 2.6 `train.py`
- CLI `--init-from PATH` → `cfg.init_from` (absolute). Required iff init_mode == 'warm'. `EXIT_INIT_MISMATCH = 6`.
- `run_pipeline` warm branch: `warm_start` after `build_model`; skip demos/BC/offline warm-up; baseline = probe of loaded actor (`bc_baseline_m`, `baseline_source='init_from'`, `init_from` dict in status); `unfreeze_at = cfg.warm.critic_warmup_steps`; provenance adds init_from (path, sha256, src steps, source rlsetup commit).
- Step counter starts at 0 (num_timesteps 0, `_n_updates` 0, reset_num_timesteps=True); lifetime steps recorded in status/provenance.
- Critic check at the first eval with step ≥ critic_warmup_steps: `critic_check = |q_min_minus_mc_nstep| / |mc_nstep_mean|` (per-step MC if nstep absent) into the eval record; > tol → `_warn` + event `critic_not_converged` (no stop).
- `PlateauStop(stop_cfg, evalcb)` after `DrivingEvalCallback`: stop when step ≥ min_steps and over the last `window` steps neither best clean det lap 2 nor best `sto_lap_mean_s` improved ≥ min_improvement_s. Event `plateau`, stop reason `'plateau'`.
- Dirty check uses `cfg.strict_provenance` (not `preset == 'full'`) in run and resume.
- Resume: `frozen_phase` for init_mode in ('bc','warm'); PlateauStop rebuilds history from evals.jsonl; init_from not re-applied.

### 2.7 `scripts/evals_to_tb.py`
Add `line/mean_abs_e_m`, `line/max_abs_e_m`, `line/share_abs_e_gt_3m`, `lap/min_edge_margin_m`, `cost/lap2_legacy`, `cost/lap2_tracking`, `cost/lap2_smoothness`, `cost/lap2_r1`, `critic/q_min_over_mc_nstep`, `diag/progress_jumps`.

## 3. Warm-start decisions
Actor loaded; critics + targets loaded (R1 value differs ≈ +4 %); log_α loaded (0.043); replay buffer fresh (relabelling only approximate: tick-1 e and e_exit not stored, stall vs off-track ambiguous; fallback if critic check fails). Critic-only phase N = 100k env steps (≈ 50k grad steps, ≈ 7 min; ≈ 25k needed, 2× margin); stochastic loaded actor acts. Unfreeze → full SAC lr 3e-4. CollapseGuard: probe every 10k for 200k, threshold 0.75 × loaded actor's probe, rollback actor_lr 1e-4, one rollback. No demos/BC.

## 4. Evaluation (same protocol as full1)
Every 50k + step 0 + end: deterministic 2-lap (lap 2 flying) + 10 stochastic 1-lap (seeds 0–9), one torch thread. **Step-0 eval must reproduce 122.84531256503526 s** (A3); it writes the new run's best.zip. Report vs full1: best clean det lap 2; median of the last 5 clean det lap 2; sto mean + clean rate; lap-2 legacy cost (vs 309.6) and cost_r1; mean/max |e|, share > 3 m, min edge margin.

## 5. Stop rule
Cap 3.0M steps (≈ 3.75 h incl. evals), max_wall_h 4.0. Plateau stop after 1.0M if neither best det lap 2 nor best sto mean improved ≥ 0.1 s in the last 500k (10 evals).

## 6. Silent breaks and fixes
| # | Where | Breaks | Fix |
|---|---|---|---|
| 1 | dirty check `preset == 'full'` | r1_warm would skip it | `strict_provenance` |
| 2 | resume `frozen_phase` only 'bc' | warm resume in critic phase unfreezes | include 'warm' |
| 3 | env `truncation` hard-coded | stall mislabelled | `_trunc_cause` |
| 4 | `laps['cost']` | would become R1 cost | keep legacy; add `cost_r1` |
| 5 | VecMonitor/TrainStats ep_return | includes shaping | document; log mean `episode_shaping` |
| 6 | `FAILURES` as "terminal" | stall no longer terminal | `TERMINAL_FAILURES` for Φ=0/P |
| 7 | CollapseGuard baseline = BC probe | no BC | loaded actor probe; 0.75 |
| 8 | demo cache key | — | already correct (key includes reward config) |
| 9 | Q-vs-MC bias band | old reward | expect step-0 Q_min − MC ≈ old bias − 7.6 − k·mean\|e\|; check tol 0.08 at unfreeze |

best.zip selection, probe progress and eval protocol are reward-independent.

## 7. Tests (fast unless marked)
1. Legacy bit-identity: fixed-seed 300-decision rollout (steer sweep, pedal steps, one forced off-track) under `ref_tracking`; sha256 of (obs, rewards, step_cost, terminated, truncated) equals a golden hash generated on commit `aeab814`.
2. Shaping exactness k=1 vs k=0, same actions: obs/dynamics bitwise equal; r(k=1) − r(k=0) == info['shaping']; Σ γ^t F_t == γ^T Φ(s_T) − Φ(s_0) rel 1e-12 for off-track (Φ_T = 0), short max_episode_s and stall (real Φ_T).
3. n-step telescoping with real SB3 `NStepReplayBuffer` (n=3, γ=0.992, n_envs=2), synthetic episodes with obs encoding Φ, reward = shaping only, mix of terminations/timeouts/cut at pos−1: `n_step_return == target_q_discount·Φ(next_obs)·(1 − final_done) − Φ(obs)`, atol 1e-4.
4. γ identity: `build_model(cfg).gamma == cfg.env.reward.gamma == DrivingEnv.cfg.reward.gamma`.
5. Stall = truncation in R1 (terminated False, truncated True, truncation 'stall', penalty 0; DummyVecEnv `TimeLimit.truncated` True); legacy still terminates with P.
6. Lap cost independent of mode and k (legacy, R1 k=0, R1 k=1 → identical `laps` and lap times).
7. Crash accounting R1: penalty == 100 independent of e; last F == −Φ(s_{T−1}).
8. warm_start on a tiny dummy env: state_dicts + log_ent_coef equal source; num_timesteps 0, buffer empty, optimizers fresh, both frozen; load_critics=False leaves critic fresh; obs-space and env-config mismatch raise `InitMismatch`.
9. Frozen phase: `train(5)` keeps actor and log_ent_coef bit-identical, critic changes; no grad step below min_train_transitions.
10. PlateauStop on synthetic histories (stop exactly when both bests stall ≥ window after min_steps; no stop if either improves ≥ 0.1 s).
11. Config: old full1 config.json loads as ref_tracking; r1_warm refuses dirty tree; r1_warm_smoke doesn't.
12. Slow: mini r1_warm_smoke from a zip of the existing slow pipeline; files written; step-0 det lap equals source; resume in frozen phase stays frozen.

## 8. Acceptance and launch
A1 tests 1–11 pass, QA clean. A2 `r1_warm_smoke` from best.zip runs, ≥ 225 env steps/s. A3 step-0 det lap 2 = 122.84531256503526 s ± 1e-6, clean. A4 actor and log_α hashes equal at step 0 and at unfreeze. A5 critic_check logged at unfreeze. A6 config.json `env.reward.mode='time_pbrs'`; provenance has init_from sha and source step 2.6M.

Launch: commit → smoke → `python -m rlsetup.driving.train --preset r1_warm --seed 0 --run-name r1_warm --init-from runs/D2/full1/seed0/checkpoints/best.zip`; `scripts/evals_to_tb.py runs/D2/r1_warm/seed0` periodically.

## 9. Risks to watch
1. More crashes as the line widens (min_edge_margin_m, det_clean, sto_clean_rate, off_track/10k).
2. Actor drift after unfreeze (critic_check, probe, rollbacks, lap 2 first 200k).
3. Projection exploits on Track.path (`progress_jumps` > 0 → sim-runner inspects `track.match` at hairpins).
4. Myopia: γ = 0.992 ≈ 5 s horizon; watch exit speeds vs QSS. γ change is a later decision.
5. Chatter without smoothness penalty (lap smoothness vs 10.4, front_lock_fraction, spin_upshifts).
6. Winner's curse in best.zip → also report median of last 5.
7. Stall truncation trend.
8. Setup-agent noise floor without a line anchor (3-seed std ≤ 0.2 s, DECISIONS #12).

## 10. Questions for Ibrahim
1. Setup loss under R1: L̄ = lap time (`cost_r1`)? (architect recommends yes). Record driver axis / z_drv as dropped or deferred in PROJECT_CONTEXT §4?
2. D2 3-seed criterion under R1: seeds 1–2 from scratch (BC + time_pbrs) or warm starts from seed 0?

## Needs web-researcher
Wiewiora 2003 (PBRS ≡ Q-init), Grześ 2017 (Φ(terminal) = 0 in episodic tasks), Ng 1999 Thm 1 wording.
