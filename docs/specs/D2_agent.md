# D2 — Driving-agent training (SAC on p0)

Status: rl-architect spec (25 Oct 2026), binding for rl-coder. Builds on DECISIONS #1–13, D1_review R10, D1_signoff (A5 obs scales and N1 fix must be merged first). SB3 2.9.0 internals read from `.venv/.../stable_baselines3` (sac.py, off_policy_algorithm.py, buffers.py, sac/policies.py, common/policies.py, torch_layers.py). Labels: [F] formulation, [C] code, [P] paper, [J] judgement.

## 0. Fixed choices
- SB3 2.9.0 SAC, CPU. γ = 0.992 per 25 Hz decision. n_steps = 3 (SB3 NStepReplayBuffer, auto when n_steps > 1).
- lr 3e-4; batch 512; buffer 1e6; τ 0.005; train_freq = 1 vector step; learning_starts = 0.
- ent_coef = "auto_0.1"; target_entropy = "auto" (= −3).
- Actor and twin critics 3×256 ReLU; critic has LayerNorm after every hidden Linear (not the output); share_features_extractor = False. use_sde = False; device = "cpu".
- From sim-runner benchmark (TrainConfig): `n_envs`, `gradient_steps`, `torch_threads` — placeholders 4, 2, 4 (UTD = gradient_steps / n_envs = 0.5).
- All step counts = transitions summed over envs (SB3 `num_timesteps`).
- Pin `stable-baselines3==2.9.0`, `sb3-contrib==2.9.0` in pyproject; test asserting the version (StagedSAC relies on 2.9.0 `train()` internals).

## 1. Module layout
```
rlsetup/driving/
  train_config.py  AgentConfig, DemoDataConfig, BCConfig, WarmupConfig, GuardConfig, EvalConfig, TrainConfig; PRESETS {'smoke','full'}
  vec.py           make_env(env_cfg_dict, rank, track_sha), build_vec_env(...). Must NOT import torch.
  policy.py        LayerNormContinuousCritic, LNSACPolicy
  agent.py         StagedSAC, build_model(cfg, vec_env), init_output_bias(actor)
  demos.py         collect_demos(...), load_demos(path), prefill_replay_buffer(rb, demos)
  bc.py            train_bc(actor, demos, cfg) -> report
  evaluate.py      flying_lap(policy, env, *, deterministic, seed, setup=None) -> LapReport
                   probe(policy, env, starts) -> ProbeReport; q_vs_mc(...); metric helpers
  callbacks.py     PhaseCallback, CollapseGuard, DrivingEvalCallback, TrainStatsCallback, WallClockLimit
  train.py         pipeline + CLI: python -m rlsetup.driving.train --preset full --seed 0 --run-name NAME [--set k=v ...]
rlsetup/runs.py    make_run_dir, write_json, update_status
rlsetup/provenance.py  (extend, §9)
scripts/collect_demos.py, scripts/d2_report.py
tests/test_agent.py, test_demos_bc.py, test_callbacks.py, test_report.py, test_d2_slow.py (slow)
```
Add `data/demos/` to .gitignore.

## 2. Vector env (vec.py)
- In `train.py`, inside `if __name__ == '__main__':`: (1) build `Track(env_cfg.track)` in main; (2) `qss = qss_profile.load_or_build(track)`; (3) `track_sha = sha256(track.path.tobytes())`; (4) `SubprocVecEnv([partial(make_env, cfg_dict, i, track_sha, qss) for i in range(n_envs)], start_method='forkserver')` wrapped in `VecMonitor`.
- `make_env` is module-level; builds `DrivingEnv(EnvConfig(**…), qss=qss)`, asserts its track.path sha == `track_sha`. QSSProfile is picklable; Track isn't.
- Seeding: `StagedSAC(seed=seed)` → `set_random_seed` seeds env i with seed + i. `torch.set_num_threads(torch_threads)` in main only.
- Every eval/probe env = `DrivingEnv(dataclasses.replace(run_env_cfg, …))`; never a fresh `EnvConfig()`.

## 3. Networks and StagedSAC (policy.py, agent.py)
**LayerNorm critic:** subclass `ContinuousCritic`; after `super().__init__`, for each idx:
```python
q = nn.Sequential(*create_mlp(features_dim + action_dim, 1, net_arch, activation_fn, post_linear_modules=[nn.LayerNorm]))
self.add_module(f"qf{idx}", q); self.q_networks[idx] = q
```
`create_mlp` adds post_linear_modules only on hidden layers. `LNSACPolicy(SACPolicy)` overrides `make_critic` to build this class with the same `critic_kwargs` (critic target picks it up via `make_critic`).

**StagedSAC(SAC)** — attributes `freeze_actor=False`, `freeze_ent_coef=False`, `actor_lr=None`:
```python
def set_frozen(self, actor: bool, ent_coef: bool):
    self.freeze_actor, self.freeze_ent_coef = actor, ent_coef
    for p in self.actor.parameters(): p.requires_grad_(not actor)

def _update_learning_rate(self, optimizers):
    super()._update_learning_rate(optimizers)          # SB3 resets all lrs each train()
    if self.actor_lr is not None:
        update_learning_rate(self.actor.optimizer, self.actor_lr)
        self.logger.record("train/actor_lr", self.actor_lr)

def train(self, gradient_steps, batch_size=64):
    if not (self.freeze_ent_coef and self.ent_coef_optimizer is not None):
        return super().train(gradient_steps, batch_size)
    opt = self.ent_coef_optimizer
    self.ent_coef_optimizer = None                      # sac.py: takes the fixed-ent_coef branch
    self.ent_coef_tensor = th.exp(self.log_ent_coef.detach())
    try: super().train(gradient_steps, batch_size)
    finally: self.ent_coef_optimizer = opt
```
Why it works [C sac.py]: with actor params `requires_grad=False`, `actor_loss.backward()` still works (`min_qf_pi` depends on the critic); Adam skips params with grad None → actor bit-identical; stray critic grads from the actor loss are zeroed by `critic.optimizer.zero_grad()` before the next critic step (standard SB3); with `ent_coef_optimizer=None` the loop uses `ent_coef_tensor`; `save()` outside `train()` sees the real optimizer, so checkpoints keep `log_ent_coef`.

**init_output_bias(actor)** (step 1): asserts `pedal_deadzone=True`; under `th.no_grad()`: `mu.weight *= 0.01`; `mu.bias = atanh([0, −0.275, −0.8])` = [0, −0.2822, −1.0986]; `log_std.weight *= 0.01`; `log_std.bias = −1`. Deterministic action ≈ [0, −0.275, −0.8] (straight, 15 % throttle, brake in deadzone), σ ≈ 0.37.

## 4. Instructive initialization (train.py)
`AgentConfig.init_mode ∈ {'bc' (default), 'bias_only' (ablation/fallback: steps 1 + 5 only)}`.

**Step 1 — output bias:** build model, `init_output_bias(model.actor)`.

**Step 2 — demos.** `DemoDataConfig`: n_transitions = 200_000; f_demo ~ U(0.35, 0.65) per episode; noise σ = 0.1 per dim i.i.d. per decision; seed = 12345 (independent of training seed); workers = 4. Per episode: `env.reset(seed=seed + ep)` (training distribution: random s0, f_env, settle) → `DemoPolicy(env, f_demo).reset()` → each step action = clip(demo + noise, −1, 1), executed and recorded → until termination/truncation (60 s) → store env reward.
Storage `data/demos/demos_<key12>.npz` + `.json` (metadata + provenance); key = sha(EnvConfig dict, DemoDataConfig, rlsetup commit, ocd_lap code version, qss cache key). Arrays: `obs` (Σ(T_i+1)×159; each episode's final obs is the true terminal/truncation obs), `actions` (N×3), `rewards`, `terminated`, `truncated`, `ep_start_obs`, `ep_start_tr`, `f_demo`, `s0`. Summary: episodes, failure causes, mean reward/decision, mean speed fraction. **Gate:** ≥ 80 % episodes without failure, else lower steering noise to 0.05 and recollect. One dataset shared across seeds.

**Step 3 — BC (bc.py).** Own Adam lr 3e-4 (not `actor.optimizer`); batch 512; ≤ 30 epochs; early stop patience 3 on val NLL; keep best; 90/10 split by episode. Loss `−mean(actor.action_dist.proba_distribution(mean, log_std).log_prob(clip(a, ±0.995)))` with `mean, log_std, _ = actor.get_action_dist_params(obs)`. Fit executed (noisy) actions → mean → clean demo action, σ → ~noise level. Report: train/val NLL, val MSE of tanh(mean) per dim, mean σ per dim, throttle-vs-rear-slip data (A1). Save `bc/actor_bc.pt`, `bc/report.json`. **Gate:** deterministic BC actor completes `flying_lap` (eval mode, f = 0.5 settle, 2 laps) without failure; record lap 2; also 3 stochastic laps (informative). If gate fails: status `bc_gate_failed`, stop; lead decides on `bias_only`.

**Step 4a — offline critic warm-up:** `prefill_replay_buffer(model.replay_buffer, demos)` (§5); `model.set_logger(configure(run/sb3, ['csv','tensorboard']))` (required before `train()` outside `learn()`; survives `learn()`); `model.set_frozen(actor=True, ent_coef=True)`; 20 × `model.train(1000, batch)`; log critic loss + mean Q on a fixed demo batch per block; end: Q vs soft MC return on demo episodes.

**Step 4b — online critic warm-up (WSRL [P]):** `model.learn(total_timesteps, callback=[...], reset_num_timesteps=True)` with `learning_starts=0` (else SB3 acts uniformly random while `num_timesteps < learning_starts` [C off_policy_algorithm.py:389]). First `online_warmup_steps = 20_000`: actor + α frozen, stochastic BC actor acts, critic updates at configured gradient_steps.

**Step 5 — unfreeze:** `PhaseCallback` at `num_timesteps ≥ 20_000` → `set_frozen(False, False)`, log event. Actor Adam fresh; α starts 0.1, learned from here. Demos leave the FIFO buffer after ~800k online transitions; no symmetric sampling.

## 5. Buffer prefill (demos.py)
Split episodes round-robin into `n_envs` columns; concatenate each column; trim to common length T. Write row by row: `rb.add(obs[n_envs], next_obs, act, rew, done, infos)` with `done = terminated or truncated`, `infos[i]['TimeLimit.truncated'] = truncated and not terminated`. Last row of every non-terminal column → `done=1`, `TimeLimit.truncated=True` (else the n-step sampler [C buffers.py:850–905] chains the cut demo into that column's first online transition). Store executed actions (action space [−1, 1], no rescale).

## 6. Evaluation (evaluate.py, callbacks.py)
- Eval/probe inference under `torch.set_num_threads(1)` (restore after); `deterministic=True` for deterministic laps. Stochastic laps save/restore `th.get_rng_state()` and use `th.manual_seed(eval_seed + k)`. Eval transitions never enter the buffer.
- Envs: `eval_det` = replace(cfg, evaluation=True, record_telemetry=True); `eval_sto` = replace(cfg, evaluation=True, eval_laps=1); `probe` = replace(cfg, max_episode_s=30).

**DrivingEvalCallback** at `num_timesteps = 0` (baseline, after online warm-up begins), every 50k, and at end; one JSON line to `evals.jsonl`:
- lap1_s, lap2_s, |lap1 − lap2|, lap-2 costs (time, smoothness, tracking, cost, smoothness share);
- mean/max |e| over lap 2 (lap index = crossings so far); min edge margin over lap 2 ≈ (1 − |track_position|)·w(s)/2 − 1.0 m, w(s) = |left − right| from `env.grid` (add to info in eval mode);
- front-lock fraction (lap-2 decisions with min(kappa_FL, kappa_FR) < −0.12, telemetry); spin upshifts (gear increase while max rear slip > 0.12);
- 10 stochastic laps: clean rate, lap times;
- Q vs MC on stochastic episodes: every 100th decision k with ≥ 600 decisions left or a failure ending: Q_min(s_k, a_k) and mean-Q vs MC_soft = Σ_j γ^j r_{k+j} − α Σ_{j≥1} γ^j log π(a_{k+j}|s_{k+j}); log mean diff and ratio;
- α, policy entropy (−mean log π), n_updates; env steps/s, grad steps/s, wall time train vs eval;
- training tallies: obs clip fraction (`vec_env.get_attr('obs_clipped_total'/'obs_values_total')`), termination-cause counts;
- saves `laps/step_{n}_det.npz` (per decision: s, x, y, speed, e, actions, gear, kappa, Fz).

**TrainStatsCallback** reads `self.locals['infos']`, `dones` each vector step; per 10k window to `train_stats.jsonl`: termination/truncation counts; speed and |e| at failure (mean; share with v < 15 m/s or |e| > 3 m); stall count among episodes with v0 < 5 m/s; episode return and length.

**CollapseGuard** (GuardConfig): probe set 8 starts, s_k = k·L/8 + 37 m, reset seed 1000 + k, `options={'start_s': s_k}`, deterministic, 30 s each (if a start raises during settle shift +50 m; save to `probe_starts.json`). BC baseline = mean probe progress of BC actor before step 4a. From unfreeze to unfreeze + 200k: probe every 10k. Pass → in-memory "last good" snapshot (deep copies of policy state_dict, actor.optimizer, critic.optimizer, log_ent_coef, ent_coef_optimizer state). Mean progress < 0.5 × BC baseline → restore snapshot, `model.actor_lr = 1e-4`, log rollback in `status.json`. Second collapse → return False (stop), status `collapsed`. Buffer never rolled back. After 200k: report only (log clean → failed deterministic eval regressions).

**WallClockLimit:** returns False past `max_wall_h`; graceful end (final eval + saves).

## 7. Checkpoints and run folder
```
runs/D2/<run_name>/seed<k>/
  config.json  provenance.json  status.json (phase, rollbacks, stop reason, timestamps)
  demos.json (dataset path, sha256, summary)   probe_starts.json
  bc/actor_bc.pt  bc/report.json
  sb3/ (progress.csv, tensorboard)   train_stats.jsonl   evals.jsonl   laps/
  checkpoints/step_<n>.zip  last.zip  best.zip (best CLEAN deterministic lap 2)
  replay_buffer.pkl (end / wall-limit stop; TrainConfig.save_replay_buffer=True; needed for D3 fine-tunes)
  stdout.log
```
Resume: `StagedSAC.load(last.zip, env=vec)` → `load_replay_buffer` → `learn(..., reset_num_timesteps=False)`.

## 8. Done-criterion report (`scripts/d2_report.py runs/D2/<run_name>`)
1. All seed folders: identical config (except seed), rlsetup commit, ocd code version — else fail.
2. Per seed re-evaluate `last.zip`, same protocol (1 thread): deterministic 2-lap run + 10 stochastic laps (eval_laps=1, seeds 0–9; clean = timed lap, no failure).
3. Report re-eval lap 2 − last logged lap 2 (expect 0; nonzero = reproducibility flag).
4. Criteria (DECISIONS #12), per seed and overall: (i) clean deterministic flying lap; (ii) lap 2 ≤ 140.6 s; (iii) plateau: lap2(eval nearest 0.8·T) − min lap2 over evals in (0.8T, T] < 0.2 s (also report lap-2 range over last 20 % as oscillation diagnostic); (iv) sample std (ddof=1) of final lap 2 across 3 seeds ≤ 0.2 s; (v) ≥ 9/10 stochastic laps clean; (vi) env steps + wall time per phase logged.
5. Write `d2_report.json` + `d2_report.md` (one table); print PASS/FAIL per criterion.

## 9. Provenance (extend rlsetup/provenance.py)
- `git_state` adds `diff_sha256` = sha256 of `git diff HEAD` (ocd_first_lap carries uncommitted Stage 6 + patch).
- Run provenance.json: `run_provenance(track)`; Python/torch/SB3/sb3-contrib/gymnasium/numpy versions; platform, hostname, `torch.get_num_threads()`, n_envs; argv, UTC start; qss cache key sha + lap time; `ReferenceLine.digest`; track sha; demo dataset sha; BC actor sha; config JSON sha.
- `full` preset refuses to run if rlsetup is dirty (`--allow-dirty` overrides, recorded). ocd_first_lap may be dirty (recorded).

## 10. Tests (fast unless marked)
- LN critic: each q-net has exactly 3 LayerNorms, all after hidden Linears; actor has none; target critic same class.
- StagedSAC (tiny numpy dummy env, obs 159, random buffer): frozen `train(5)` keeps actor params + `log_ent_coef` bit-identical while critic + target change; unfrozen all change; `actor_lr` override works while critic stays 3e-4; save/load round-trip → identical deterministic actions; SB3 version == 2.9.0.
- Output bias: deterministic action on zero obs ≈ [0, −0.275, −0.8] ± 0.01; log_std ≈ −1.
- Demos: schema; terminal next_obs is true final obs; recorded reward == env reward; same seed → byte-identical (2 short episodes, max_episode_s = 2).
- Prefill: obs encode episode id; 1,000 n-step samples never mix episodes; column tails flagged.
- BC: synthetic a = tanh(Wx) + N(0, 0.1) → val MSE < 0.05, σ in expected range.
- CollapseGuard: fake probe values → rollback restores state dicts + sets actor_lr; second collapse stops.
- vec.py: importing in fresh `python -I -c` doesn't load torch; worker with wrong track sha raises.
- Metric helpers: Q-vs-MC on synthetic rewards (exact), lap-index tagging, front-lock and spin-upshift counters, edge-margin formula.
- Report: synthetic `evals.jsonl` for 3 seeds → PASS and FAIL for each criterion.
- N1: `setup=None` falls back to p0.
- Slow: end-to-end mini pipeline (demos 5k, BC 2 epochs, offline 200, online 3k with 2 workers, 1 eval, 1 probe); all files written; resume from `last.zip` + buffer works.

## 11. First-run plan
0. Prereqs: A5 + N1 merged; sim-runner benchmark fills `n_envs`, `gradient_steps`, `torch_threads` (pick by lap time per wall-hour among UTD ∈ {0.25, 0.5, 1}); demos collected once (~2–4 min), gate passed.
1. **Smoke, ~20 min, seed 0** (`--preset smoke`): demos as above; BC ≤ 10 epochs + gate; offline 5k; online warm-up 10k; RL until 20 min wall; probe every 5k; eval every 25k with 3 stochastic laps. Pass if: no exceptions; throughput within 20 % of benchmark; critic loss finite and |Q| < 2,000; after unfreeze probe progress ≥ 0.5 × BC (or one rollback that recovers); α in [1e-4, 1]; obs clip < 1 %; checkpoint save/load/resume works; report script runs (criteria FAIL, no crash). Extrapolate steps/h → `total_timesteps` for full.
2. **Full:** seeds 0, 1, 2 sequentially (each uses the performance cores). Placeholder `total_timesteps` = 2e6 until smoke extrapolation, `max_wall_h` per seed. Run d2_report. Only (iii) fails → resume all seeds +50 %. (ii) fails clearly at plateau → escalate (questions below).

## Risks (metric)
1. BC collapse after unfreeze (probe progress vs BC baseline; rollbacks; Q jumps). 2. Termination exploitation (failure share rising with return; share at v < 15 m/s or |e| > 3 m). 3. Overestimation (Q_min − MC_soft growing). 4. Stuck at BC pace (demos f ≤ 0.65): lap-2 trend; if > 200 s at 500k revisit demo f range or entropy target. 5. Front lock / gear hunting. 6. Chatter (smoothness share). 7. α collapse/explosion (α, entropy vs −3). 8. Seed spread > 0.2 s (setup agent's noise floor). 9. Eval irreproducibility (re-eval delta). 10. Speed vs tracking under c = 0.1 (mean |e| rising as lap time drops). 11. Compute (grad vs env steps/s). 12. nonfinite > 0 → sim-runner.

## Questions for Ibrahim
1. (OPEN §11.8, contingent) If D2 plateaus above 140.6 s mainly because it spends lap time staying near ξref (high tracking share, low |e|): lower c for the base driver, or relax the 1.05× target? Nothing needed now.
2. Compute budget: wall-clock hours on the M4 for the 3-seed full run (sets `max_wall_h`), or the lab machine?
