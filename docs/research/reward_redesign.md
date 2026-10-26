# Reward redesign for the SAC driving agent: research note

Prepared by `web-researcher`, 26 Oct 2026. Context: SAC driver in OCD, F1-like car, 25 Hz decisions (action repeat 2 over 50 Hz ticks), absolute actions [steer, throttle, brake], γ = 0.992, ξref = safe minimum-curvature line (DECISIONS #1, #6, #13). Current per-tick reward
r = a·Δs − b·Δt·(1 + u̇ᵀWu̇) − c·e², with e = distance to ξref; off-track / stall / wrong-way terminate with P = 50 + 30·c·e². Goal stated by the lead: the fastest possible driver, loosely tied to ξref.

**How to read this.** Plain statements and "Source says" bullets report what the cited paper or repo file says (section, table, file, commit). **(derived)** marks arithmetic I did on the source's numbers. Opinions appear only under **Interpretation**. I say where I could not open a source.

---

## 0. Sources opened today

| Key | Source | How accessed |
|---|---|---|
| LS | Linesight (Trackmania RL), github.com/Linesight-RL/linesight @ `f6e7822fcb91d10bafd9b8db070a1dee78046558` (30 Sep 2026) | raw files: `config_files/config.py`, `trackmania_rl/buffer_management.py`, `trackmania_rl/buffer_utilities.py`, `trackmania_rl/agents/iqn.py`, `trackmania_rl/tmi_interaction/game_instance_manager.py`, `trackmania_rl/reward_shaping.py`, `README.md`, `docs/source/{main_objects,first_training,second_training,user_faq}.rst`; docs site index |
| ALT99 | E. Altman, *Constrained Markov Decision Processes*, Chapman & Hall/CRC, 1999, ISBN 0849303826 | bibliographic record only (book not opened) |
| RCPO | Tessler, Mankowitz, Mannor, "Reward Constrained Policy Optimization", ICLR 2019, arXiv:1805.11074 | ar5iv full text |
| PID | Stooke, Achiam, Abbeel, "Responsive Safety in RL by PID Lagrangian Methods", ICML 2020, arXiv:2007.03964 | ar5iv full text |
| HA20 | Ha, Xu, Tan, Levine, Tan, "Learning to Walk in the Real World with Minimal Human Effort", arXiv:2002.08550 (arXiv page lists no venue) | ar5iv full text |
| WCSAC | Yang, Simão, Tindemans, Spaan, "WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained RL", AAAI 2021, 35(12):10639–10646, doi:10.1609/aaai.v35i12.17272 | abstract only (PDF and Springer journal version not readable/paywalled) |
| SPOOR25 | Spoor, Serra-Gómez, Plaat, Moerland, "Towards a Practical Understanding of Lagrangian Methods in Safe RL", arXiv:2510.17564v2 (Mar 2026) | arXiv HTML |
| TRAD26 | Leng et al., "Expert Knowledge-driven RL for Autonomous Racing via Trajectory Guidance and Dynamics Constraints", arXiv:2603.05842v3 (Aug 2026) | arXiv HTML (first 100k of 118k chars) |
| APEX26 | Zhou et al., "Push the Limits: Human-Like Multimodal Extreme Embodied Intelligence for Autonomous Racing", Advanced Robotics Research (23 Oct 2026), doi:10.1002/adrr.202500210 | Wiley Scholar Gateway passages (Methods) only |
| YU25 | Yu, Fu, Zhang, Yang, "Enhancing Safety in Autonomous Racing With Constrained RL", IEEE RA-L 10(6):6448–6455, 2025, doi:10.1109/LRA.2025.3566591 | abstract only |
| FUCHS21 | Fuchs et al., "Super-Human Performance in Gran Turismo Sport Using Deep RL", RA-L 2021, arXiv:2008.07971v2 | arXiv HTML |
| LEE25 | Lee et al., "A Champion-level Vision-based RL Agent for Competitive Racing in GT7", RA-L 2025, arXiv:2504.09021 | arXiv HTML |
| TRI24 | Subosits et al., "Towards an Autonomous Test Driver", arXiv:2412.03803 | arXiv HTML |
| ACGYM | Remonda et al., Assetto Corsa Gym, repo `dasGringuen/assetto_corsa_gym` @ `ef0fa502` | `config.yml`, `assetto_corsa_gym/AssettoCorsaEnv/ac_env.py` |
| SOPHY22 | Wurman et al., Nature 602 (2022), doi:10.1038/s41586-021-04357-7 | **not re-opened today** (PDF not parseable, Nature page behind login). Statements come from our 25 Oct note `driving_agent_sota.md`, which read the full text. |
| WANG24 | Wang, Zheng, Lin, "Steady-State Error Compensation for RL with Quadratic Rewards", arXiv:2402.09075v2 | arXiv HTML |
| CAPS21 | Mysore, Mabsout, Mancuso, Saenko, "Regularizing Action Policies for Smooth Control with RL", ICRA 2021, arXiv:2012.06644 | abstract |
| CHRIST24 | Christmann, Luo, Mandala, Chen, "Benchmarking Smoothness and Reducing High-Frequency Oscillations in Continuous Control Policies", IROS 2024, arXiv:2410.16632 | arXiv HTML |
| SEYDE21 | Seyde et al., "Is Bang-Bang Control All You Need?", arXiv:2111.02552 | abstract |
| RUDIN21 | Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep RL", CoRL 2021, arXiv:2109.11978; repo `leggedrobotics/legged_gym` @ `8fa29acc` (29 May 2025) | repo files `legged_gym/envs/base/legged_robot_config.py`, `legged_robot.py` |
| PARDO18 | Pardo, Tavakoli, Levdik, Kormushev, "Time Limits in RL", ICML 2018, arXiv:1712.00378 | abstract |
| KNOX | Knox, Allievi, Banzhaf, Schmitt, Stone, "Reward (Mis)design for Autonomous Driving", arXiv:2104.13906v2 | arXiv HTML (first 100k chars) |
| Libraries | PyPI JSON for `omnisafe`, `safety-gymnasium`, `fast-safe-rl`, `tianshou` 0.5.0; GitHub API for PKU-Alignment/omnisafe, liuzuxin/FSRL, openai/safety-starter-agents; OmniSafe @ `15603dd7` files `omnisafe/algorithms/off_policy/sac_lag.py`, `omnisafe/common/lagrange.py`, `omnisafe/configs/off-policy/SACLag.yaml`, `pyproject.toml`, `README.md`; local SB3 2.9.0 `stable_baselines3/sac/sac.py` | direct |

Could not open: the SMC 2020 paper "Two-Stage Safe RL for High-Speed Autonomous Racing" (meridiancas.github.io PDF, binary not parseable; title/authors unverified), the WCSAC full text, the Safety Gym paper (Ray et al. 2019).

---

## 1. Linesight (Trackmania, IQN)

All file references are to commit `f6e7822f`.

### 1.1 Source says

**Reward per transition** (`config_files/config.py`; computed in `buffer_management.py::fill_buffer_from_rollout_with_n_steps_rule`):

```python
constant_reward_per_ms = -6 / 5000
reward_per_m_advanced_along_centerline = 5 / 500
final_speed_reward_as_if_duration_s = 0
shaped_reward_dist_to_cur_vcp = -0.1
shaped_reward_min_dist_to_cur_vcp = 2
shaped_reward_max_dist_to_cur_vcp = 25
shaped_reward_point_to_vcp_ahead = 0
engineered_speedslide_reward_schedule = [(0, 0)]
engineered_neoslide_reward_schedule = [(0, 0)]
engineered_kamikaze_reward_schedule = [(0, 0)]
engineered_close_to_vcp_reward_schedule = [(0, 0)]
```

- `reward_into[i]` = `constant_reward_per_ms` × elapsed ms of the step (50 ms; the final frame gets only the remaining race time) + `reward_per_m_advanced_along_centerline` × change in `meters_advanced_along_centerline` (`buffer_management.py`).
- So the active reward is **progress along the reference line minus a constant time cost**. All "engineered" terms (speedslide, neoslide, kamikaze, close-to-VCP) are scheduled at 0. `final_speed_reward` is 0 because `final_speed_reward_as_if_duration_s = 0`.
- **Line-distance term: only as potential-based shaping (PBRS).** `get_potential()` (`buffer_management.py`) returns `shaped_reward_dist_to_cur_vcp × clip(‖state_float[62:65]‖, 2, 25)`, i.e. −0.1 × the clamped **3-D Euclidean distance to the current virtual checkpoint** (the next reference point), not a lateral offset. It is applied at batch time in `buffer_utilities.py::buffer_collate_function`: `rewards += np.where(terminal, 0, gammas * next_state_potential)` and `rewards -= state_potential`, i.e. r_n + γⁿΦ(s′) − Φ(s) with Φ(terminal) = 0. `docs/source/main_objects.rst` cites Ng et al.'s potential-based shaping for this.
- **Control-smoothness term: none.** No term on action changes. Actions are discrete (12 combinations of accelerate/brake × left/none/right, `config_files/inputs_list.py`, per our 25 Oct note). Previous 5 actions are in the input (`n_prev_actions_in_inputs = 5`).
- **The reference line.** Virtual checkpoints are points every 0.5 m along a replay trajectory (`distance_between_checkpoints = 0.5`). `docs/source/second_training.rst`: "This reference trajectory does not need to be fast." The authors usually drive the map centerline.
- **How loosely the car is tied to the line.** The zone index advances to the next reference point only while it is closer and within `max_allowable_distance_to_virtual_checkpoint = sqrt((distance_between_checkpoints/2)**2 + (road_width/2)**2)` with `road_width = 90  ## a little bit of margin, could be closer to 24 probably ?` (config.py). That is ≈ 45 m **(derived)**. Index can also step backwards (`game_instance_manager.py::update_current_zone_idx`).
- **Crashes, going off the line, timeouts.** No wall-contact check, no off-line check, no penalty (`game_instance_manager.py`). A rollout is cut when `_time > last_progress_improvement_ms + self.max_minirace_duration_ms` (constructor default 2000 ms; config key `cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_000`; I did not trace the call site that passes it) or when the overall cap is exceeded (`cutoff_rollout_if_race_not_finished_within_duration_ms = 300_000`). On a cutoff, `rollout_results["race_time"]` is **not** set, so `terminal_actions = math.inf` (`buffer_management.py`): **cut-off rollouts are treated as truncation and bootstrapped, not as terminal states.** Only finishing the race is terminal.
- **Discount / horizon: "temporal mini-race".** `temporal_mini_race_duration_ms = 7000`; `gamma_schedule = [(0, 0.999), (1_500_000, 0.999), (2_500_000, 1)]`; `n_steps = 3`. At batch time a random mini-race clock is sampled, written into `state_float[:, 0]`, n-step windows are shortened at the mini-race end, and a transition is terminal when `next_time >= temporal_mini_race_duration_actions` (`buffer_utilities.py`). `main_objects.rst`: this lets the agent be optimized with gamma = 1; Q is "the sum of expected rewards obtained during the next 7 seconds".
- **Algorithm.** IQN with dueling heads (`agents/iqn.py`): `iqn_n = 8`, `iqn_k = 32`, `iqn_kappa = 5e-3`, `iqn_embedding_dimension = 64`, `use_ddqn = False`, `prio_alpha = 0`. CNN on a grayscale image + float MLP (`float_hidden_dim = 256`), `dense_hidden_dimension = 1024`. `batch_size = 512`; `lr_schedule` 1e-3 → 5e-5 (3M steps) → 1e-5 (15M); `soft_update_tau = 0.02`; ε-greedy 1 → 0.1 (300k) → 0.03 (3M), plus Boltzmann ε 0.15 → 0.03. `discard_non_greedy_actions_in_nsteps = True`.
- **Observation** (`game_instance_manager.py`): mini-race time slot, previous 5 actions, per-wheel sliding/contact/damper/material, gear and RPM, car-frame angular and linear velocity, 40 reference points in the car frame (every 20th 0.5 m checkpoint, so 400 m ahead **(derived)**), distance to finish (capped), freewheel flag; plus a 1-channel image.
- **Control rate.** `ms_per_action = 10 ms × 5 = 50 ms` → 20 Hz.
- **Budget and results.** `first_training.rst`: presses forward by ≈300k steps, first finishes ≈500k, finishes regularly ≈1M; the early phase is the first 3–5M steps; then "low 54 seconds" on ESL-Hockolicious. Total budget for the record runs is not stated in the repo. Hardware tested (user FAQ): Ryzen 7 5700G, 64 GB, RTX 4070 Ti; 4 game instances optimal on Linux. Docs index: "Our AI was able to beat ten out of twelve prestigious world records in the official Trackmania campaign" (tool-assisted; README: first AI to beat world records on official campaign tracks, May 2024).
- **On time-optimality / shaping.** The repo documentation does not discuss time-optimality explicitly. The design facts are the ones above: progress + constant time cost, PBRS only, no engineered terms in the shipped config.

### 1.2 Derived numbers
- Reward rates: +0.01 per metre, −1.2 per second. Per-step reward is positive only above 120 m/s (432 km/h) **(derived)**, so it is almost always negative.
- Within a fixed 7 s mini-race with γ = 1 and no early termination, the time term sums to −8.4 for every policy except when the race finishes inside the window **(derived)**. The return being maximized is therefore 0.01 × (metres gained along the reference in the next 7 s), plus a bonus for finishing earlier near the end.

### 1.3 Interpretation
- Linesight's "fastest driver" recipe: measure progress by projecting onto a reference that "does not need to be fast", give no lateral penalty, shape only with PBRS (policy-invariant per Ng et al.), never terminate on crashes, and use a fixed undiscounted 7 s window with the clock in the state (consistent with PARDO18's "time-awareness" argument). The constant time cost cannot create a termination incentive because nothing but the finish terminates.
- The line it uses is a measuring tape, not a driver. That is the opposite of our need, where ξref **defines** the driver. Linesight's design is evidence for how to make the progress/time part unbiased, not for how to keep the driver near ξref.

---

## 2. Constrained MDPs and Lagrangian methods

### 2.1 Source says: foundations
- **ALT99**: the CMDP book (finite state space, one controller with several competing objectives, per the library record). Not opened.
- **RCPO** (ICLR 2019):
  - Lagrange relaxation with a multiplier λ on a slower timescale: "λ is increased until the constraint is satisfied" (Sec. 3). Three timescales: critic, actor, λ (Sec. 4.2, Alg. 1). Penalized reward r̂ = r − λc (Def. 3, Eq. 10) so a single critic can learn it; λ is updated from Monte-Carlo estimates of the constraint.
  - Experiments: Mars rover (A2C; λ₀ = 0.6, λ lr 2.5e-5) and MuJoCo with an average-torque constraint below 25 % (PPO; λ₀ = 0, λ lr 5e-7, decayed by κ = 1 − 1e-9 "in order to avoid oscillations", App. B.2).
  - On fixed penalties: "Selecting a constant coefficient λ such that the policy satisfies the constraint is not a trivial task" (Sec. 5.2.3) and the coefficient "is not shared across domains" (Sec. 1).
  - Table 2 is mixed: RCPO is feasible in most domains, but in Walker2d λ = 0 earned far more reward (3364 vs 592), and in Hopper the authors say the better method is unclear.
  - Guarantees are local: convergence to "a fixed point (a local minima) almost surely" (Thm. 1), assuming every local minimum of the constraint is feasible (Assumption 2). Without it the constraint acts "as a regularizing term" (App. E).
- **PID** (ICML 2020):
  - Plain gradient-ascent multipliers: "their learning dynamics exhibit oscillations and overshoot" (abstract); "intermediate iterates often violate constraints" (Sec. 1). The standard update "amounts to integral control on the constraint" (Sec. 1).
  - Fix: λ = (K_P·Δ + K_I·I + K_D·∂)₊ with Δ = J_C − d, I ← (I + Δ)₊, ∂ = (J_C − J_C,prev)₊ (Alg. 2). Gains used: K_P = 1 or 0.25, K_I swept 1e-4 to 1e-1, K_P = 0.1 with K_I = 1e-3 in Fig. 9; K_D not stated numerically. Tuning was manual, with smoothing on P and D terms (Sec. 6.2).
  - Setting: Safety Gym (Point/Car/Doggo), PPO with an LSTM policy, cost limits 25–200 per 1000-step episode, ≈40M steps. Results are in figures only: PI "damps oscillations, dramatically reducing violations"; D "removed cost overshoot" (Sec. 6.3).
  - Reward-scale invariance: scale the cost gradient by β = ‖∇J‖/‖∇J_C‖ (Eq. 26); dynamics "nearly identical across two orders of magnitude of reward scale" (Fig. 9).

### 2.2 Source says: SAC with Lagrangian constraints
- **HA20** (real Minitaur, SAC):
  - Safety function f_s = min(p̂ − |p|, r̂ − |r|) on torso pitch/roll, thresholds π/12 and π/6 (Sec. 4.2, Eq. 4), constraint E[f_s] ≥ 0 "∀t" (Eq. 3).
  - Separate safety critic S_ψ; actor loss −Q − λS_ψ (Eq. 6); λ by dual gradient descent (Eq. 7), Adam lr 3e-4, alongside SAC's α (two multipliers). About 50 Hz control, 2×256 MLPs, last 6 actions in the observation, actions low-pass filtered.
  - Sim result (Sec. 5.3, 5 seeds): unconstrained SAC "falls over 100 times", the Lagrangian version ≈40 times, **more than a well-tuned fixed weight of 1.0**. A fixed weight of 100 was excessive.
- **WCSAC** (AAAI 2021, abstract only): SAC + a distributional safety critic with CVaR. Motivation: "it can be hazardous to set constraints on the expected safety signal without considering the tail of the distribution". The safety measure adapts the reward/safety trade-off weights.
- **SPOOR25** (2025–26, Omnisafe, mainly PPO-Lag, one SAC-Lag preliminary):
  - "Our results reveal the highly sensitive nature of λ" (abstract). The optimal fixed λ* is "highly task-dependent" (App. A.1).
  - Gradient ascent (η = 0.035) vs PID (K_P = K_I = 1e-4): "no single method consistently performs best across or within tasks" (Sec. 6).
  - Lagrangian methods "are not strict constraint enforcers by design", so practitioners may need stricter training limits (Sec. 6). Several runs violated limits (Table 1).

### 2.3 Source says: racing papers with Lagrangian constraints (2019–2026)
- **TRAD26** (arXiv 2026, the closest racing example):
  - Lagrangian costs on **vehicle stability, not track limits or line distance**:
    - yaw-rate margin h_ω = μg/u − |ω| (Eq. 23);
    - sideslip margins against a Pacejka-based bound (Eqs. 21–26);
    - written as CBF-style costs c = max(−(ḣ + αh), 0), averaged over a 40-step window.
  - Multipliers by dual ascent, lr 5e-5 vs actor/critic 1e-4 (Sec. 4.1). Penalty applied only when predicted cost exceeds d_k: a "safety dead-zone" (Sec. 3.4, Eq. 31). The cost limits d_k are not given numerically, and λ trajectories are not reported.
  - Track limits stay in the reward (+1 on track / −2 off, Eq. 7), and line tracking too (1/(1+l²), weight 0.6, Eq. 17). No action-smoothness term: oscillations "will be further addressed" (Sec. 5).
  - Results:
    - Constraints cost lap time. On Berlin, 46.05 s without the dynamics constraints vs 53.94 s with them, with β-violation time down 48.75 % (Table 4).
    - The **SAC-Lag baseline was much more conservative**: 104.63 s, average speed 23.07 m/s (Table 3), which the authors attribute to speed being traded for safety (Sec. 4.8).
    - "cannot guarantee strict constraint satisfaction at every time step" (Sec. 4.8).
  - Simulator: an unnamed simplified dynamics model with a linear tyre model.
- **APEX26** (real super-formula car, Yas Marina; Methods passages only):
  - SAC-like "constrained policy iteration" with two Lagrange multipliers: C₁ an action-related cost bounded above, C₂ the policy entropy bounded below (Eqs. 15–16).
  - Reward penalizes previewed deviation from the reference trajectory.
  - Proportional longitudinal/lateral controllers assist early training and act as a test-time fallback.
  - Threshold values not visible.
- **YU25** (RA-L 2025, abstract only): racing as a CMDP with "two constrained RL algorithms" (not named in the abstract) plus a rollover-dynamics shield limiting speed. F1TENTH sim and a 1/10 car capped at 4 m/s.
- **Two-stage safe RL for high-speed racing (SMC 2020)**: a search snippet says safety is a single constraint on the minimum distance to the two road boundaries, with DDPG-based variants. **Not opened; title/venue unverified.**
- I found **no paper that puts distance to a racing/reference line as a Lagrangian constraint** for an RL racing driver (two web searches, one Scholar Gateway search).

### 2.4 Known pitfalls (sources above)
1. Oscillation/overshoot of λ and constraint violations during training with gradient-ascent multipliers (PID Sec. 1; SPOOR25 Sec. 4.2). Mitigations: slow λ lr, PI(D) control, and RCPO's decaying λ lr.
2. λ is highly sensitive and task-dependent, and no update rule wins everywhere (SPOOR25).
3. Constraints are met only in expectation and only at a local saddle (RCPO Thm. 1). Expectation constraints ignore tails (WCSAC).
4. Learned λ can be worse than a well-tuned fixed weight (HA20 Sec. 5.3; RCPO Table 2 Walker2d).
5. In racing, SAC-Lag became conservative and slow (TRAD26 Table 3).

---

## 3. Implementations for PyTorch / SB3 2.9.0 / Gymnasium

| Library | Latest | Status | SAC-Lag? | Fit with our stack |
|---|---|---|---|---|
| **OmniSafe** (PKU-Alignment) | PyPI 0.5.0, 2024-05-04 | not archived; last push 2025-03-17 (`15603dd7`, README edit); Apache-2.0; 29 open issues | SACLag, SACPID, TD3Lag/PID, DDPGLag/PID, plus on-policy PPO-Lag, CPPO-PID, RCPO | `pyproject.toml` classifiers stop at Python 3.11. Depends on `safety-gymnasium>=0.2.0`, whose latest (1.0.0, 2023-07-21) pins `gymnasium==0.28.1`, `mujoco==2.3.3`, `pygame==2.1.0`. It also pins `pandas==2.0.3`, `scikit_learn==1.3.2` and others. Our venv has gymnasium 1.4.0 / Python 3.12, so a conflict is expected (not tested). Custom envs go through its own CMDP interface. |
| **FSRL** (liuzuxin) | PyPI `fast-safe-rl` 0.1.0, 2023-06-14 | last commit 2024-09-30 (`e056fc94`, "Update setup.py"); MIT | SACLagrangian with a **PID** Lagrangian; multiplier "updated on-policy" (README) | Built on Tianshou `~=0.5.0` (2023); pins `wandb~=0.14.0`, `protobuf~=3.19.0`, needs `bullet-safety-gym`. Effectively unmaintained. |
| **safety-starter-agents** (OpenAI) | n/a | **archived**; last push 2023-04-02 | PPO/TRPO-Lag, CPO (Safety Gym baselines) | TensorFlow 1 era; reference only. |
| **SB3 / sb3-contrib** | 2.9.0 | active | **No** constrained SAC in SB3, sb3-contrib or SBX (web search found none) | — |

**OmniSafe SACLag internals** (`@15603dd7`):
- λ is updated once per epoch, after `warmup_epochs`, from the logger's episodic cost: `Jc = self._logger.get_stats('Metrics/EpCost')[0]` (`sac_lag.py`).
- Update loss: `-lambda * (mean_ep_cost - cost_limit)`, then clamp to [0, upper bound] (`common/lagrange.py`).
- Actor loss: (α·logπ − min Q_r + λ·Q_c) / (1 + λ).
- Defaults (`SACLag.yaml`): `cost_limit 25`, `lagrangian_multiplier_init 0.001`, `lambda_lr 1e-5` (Adam), `alpha 0.2` fixed, γ 0.99, batch 256, 2×256 MLPs.
- Safety-Gym overrides: `lambda_lr 5e-7`, actor lr 5e-6, `alpha 1e-5`. SafeMetaDrive: `cost_limit 0`, `lambda_lr 1e-4`.

**What SB3 2.9.0 SAC would need** (read from local `.venv/.../stable_baselines3/sac/sac.py`):
- `train()` already optimizes one dual variable: `ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()`.
- The actor loss is `(ent_coef * log_prob - min_qf_pi).mean()`.
- Targets use `replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values`, with n-step discounts supported.
- A SAC-Lag subclass would need three additions:
  1. a cost signal stored per transition (a ReplayBuffer subclass with a `costs` field, or an n-step-aware equivalent);
  2. a second twin critic Q_c with its own target;
  3. a λ parameter updated from episodic or batch cost, plus the term λ·Q_c in the actor loss.
- **Interpretation:** about 150–300 lines, mirroring the existing α code. An RCPO-style shortcut (one critic on r − λc) needs r and c stored separately, because stored rewards go stale when λ changes.

---

## 4. Penalty shape (L1 / L2 / Huber / deadband) and action-rate penalties

### 4.1 Source says: line-deviation penalty shapes in practice

| Agent | Line / edge term | Shape |
|---|---|---|
| LS | none (PBRS on distance to next checkpoint, clamped 2–25 m) | — |
| FUCHS21 | none; wall contact −c_w‖v‖² | — |
| SOPHY22 (25 Oct note) | none on a line; off-course −Δt·v_kph², wall same form | speed² penalty while off course |
| ACGYM `ac_env.py` | r = v_kmh·(1 − \|gap\|/12)/300 (hard-coded 12.00, not `gap_const`) | **L1, multiplicative with speed** |
| TRI24 | R_edge = q₁·e_norm·v² when \|e_norm\| > 1; optional R_imi = −(y − μ_human)²/max(σ, σ_min)² | **hinge outside the track**; imitation term **variance-normalized L2** |
| TCD23 (25 Oct note) | progress − \|n\| | L1 |
| TRAD26 | 1/(1 + l²) reward, weight 0.6 | bounded, Cauchy-like |
| RUDIN21 / legged_gym | tracking = exp(−err²/σ), `tracking_sigma = 0.25` | Gaussian kernel (bounded) |

- **TRI24 on path imitation** (Sec. on R_imi, Table II): it incentivizes "the agent to position the car precisely where the human is very consistent". Average lap times: pure RL 96.69 s, augmented reward 96.42 s, augmented + regularization 96.54 s. The authors note these differences are small relative to seed variation.
  - **Correction to `driving_agent_sota.md` §3**: that note said the imitation term "cost ≈ 0.1–0.2 s". Table II shows it was **slightly faster**, not slower.
- **WANG24** (DDPG, 0.1 s steps; ACC and a bicycle-model lane change, single runs):
  - Quadratic rewards leave steady-state error: lane change 2.3e-2 m vs 5.5e-5 m for absolute-value (Table IV).
  - Absolute-value rewards show "larger fluctuations in the wheel angle change rate" (Sec. IV-B3).
  - Quadratic was chosen "for its superior smoothness" (Sec. III-B); an integral term reduces the quadratic error.
- **I found no racing-specific ablation of L1 vs L2 vs Huber/deadband line penalties** (two web searches). That is an absence of evidence.

### 4.2 Source says: action-rate / smoothness penalties in top racing agents

| Agent | Smoothness in reward? | Other smoothing |
|---|---|---|
| FUCHS21 | **No** (Eq. 1 = progress − wall penalty) | Evaluation-only rate caps (0.03 rad/frame steering, 80 %/frame pedals) "not significantly influenc[ing]" lap times (Sec. IV). Jitter appears only with observation noise: "More noise in the agent's observations leads to jitter in its action signal, resulting in a speed loss" (Sec. V-C). Without a wall penalty, policies "did not brake and simply grinded along the track's walls" (Sec. III-A). |
| SOPHY22 (25 Oct note) | **No** action-rate term in the time-trial reward listed there | 10 Hz control |
| LEE25 (GT7, vision) | **Yes**: steering change r^s = −\|θ_t − θ_{t−1}\| (weight 0.5); steering-history penalty on sign flips of consecutive steering deltas above c^d = 0.014 (weight 5.0) | delta steering ±3° per 10 Hz step; steering linearly interpolated between updates (Sec. III-B/C) |
| TRI24 | **Yes**: "linearly penalize steering rates and changes in longitudinal input beyond the maximum observed in human demonstrations"; throttle release and brake application are not penalized, "to allow as rapid of a transition from gas to brake ... as possible" | 10 Hz |
| ACGYM | Optional `penalize_actions_diff` (L2 norm, not squared, × 0.1), **off by default** | delta actions, `max_steer_rate: 900.0`; deltas "greatly reduce oscillation" (paper Sec. 4.2) |
| LS | No | discrete actions, 20 Hz |
| TRAD26 | No; oscillation left to future work | — |
| legged_gym (non-racing reference) | `action_rate = -0.01` × Σ(a_t − a_{t−1})² | — |

- **legged_gym** also clips the total reward, with the stated reason: `only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)`.
- **Generic evidence:**
  - SEYDE21: trained continuous-control agents often pick actions at the bounds (bang-bang); Bernoulli bang-bang policies match SOTA on benchmarks.
  - CAPS21: action-policy regularization removes high-frequency control components; ≈80 % less power on a real quadrotor.
  - CHRIST24, Sec. I, on reward penalties: "A straightforward way to mitigate the issue is to include penalization terms as part of the reward function", but the agent may prefer subpar performance, and the weight is hard to tune (too large → policy does little, too small → ignored). Its best hybrid (LipsNet + L2C2) improved FFT smoothness by 26.8 % with ≤ 2.8 % worst-case performance loss. Low-pass filters have memory, which breaks the Markov property unless history is observed.
  - KNOX lists "unsafe reward shaping" and "learnable loophole(s)" among 8 sanity checks for driving rewards. Shaping can lower the performance ceiling "by changing the reward function's preference ordering over policies" (Sec. 4.1).

### 4.3 What happens without one (sources)
- The fastest racing agents (FUCHS21, SOPHY22, Linesight) reached superhuman lap times **without** an action-rate reward term. FUCHS21 shows rate caps barely affected lap time. Jitter costs speed mainly when observations are noisy.
- Agents that added smoothness terms either use high-capacity vision policies (LEE25) or aim at human-like test driving (TRI24). TRI24 deliberately exempts the throttle-to-brake transition.

---

## 5. Implications for us (interpretation, not source claims)

1. **Progress/time core.**
   - Linesight and FUCHS21 support keeping a·Δs − b·Δt as the speed objective.
   - Linesight additionally avoids any termination incentive: there is no termination except the finish, a fixed undiscounted window, and the clock in the state.
   - Our terminating episodes with a negative per-step reward keep the trap noted on 25 Oct. legged_gym's `only_positive_rewards` comment flags the same issue.
   - A cheap check is that P exceeds the avoidable discounted living cost (≈ b·Δt·(1+max u̇ᵀWu̇)·125 steps at γ = 0.992, derived). The other option is to treat stall cut-offs as truncation (bootstrap), as Linesight does.
2. **"Loosely tied" needs a non-PBRS term, but its shape matters.**
   - c·e² is flat near the line and steep far away. It allows small offsets almost for free (WANG24's steady-state-error effect) but punishes large line changes quadratically.
   - For "free inside a corridor, firm outside", the evidence-backed shapes are:
     - a deadband/hinge (TRI24's edge and rate terms are hinges);
     - a σ-normalized L2 that is soft where the reference is uncertain (TRI24's R_imi);
     - a bounded kernel (legged_gym, TRAD26).
   - There is no racing ablation, so a small A/B (e², Huber with deadband d, hinge) is ours to run.
3. **Lagrangian constraint as the alternative to a fixed c.**
   - A CMDP formulation: maximize progress − time s.t. E[mean e²] ≤ d, or time-fraction outside a corridor ≤ δ.
   - It would make the tie a tolerance (d) instead of a price (c), and λ would adapt per setup.
   - That conflicts with "c is constant and the same for all drivers" (PROJECT_CONTEXT §4) and touches OPEN item §11.8 (c). This is **Ibrahim's call**.
   - Risks from the sources: λ oscillation and violations (PID), λ sensitivity (SPOOR25), conservative slow SAC-Lag in racing (TRAD26), and only expected-value satisfaction.
   - If adopted: a PI-controlled λ on episodic cost, a slow λ lr, and a stricter training limit than the target.
4. **Smoothness term.**
   - The fastest agents had none. A quadratic u̇ᵀWu̇ on **pedals** penalizes the throttle-to-brake snap that time-optimal driving needs, which is exactly what TRI24 exempts.
   - Options, cheapest first:
     - keep W small, or W on steering only;
     - hinge on rates above a physical limit (TRI24);
     - an env-side steering-rate limit (ACGYM/FUCHS21);
     - CAPS/L2C2-style policy regularization (keeps the reward clean).
   - Whether u̇ stays inside the lap cost L̄ (Limebeer form) is a formulation question for Ibrahim.
5. **Library.**
   - No drop-in SAC-Lag for SB3 2.9.0 / Python 3.12 / gymnasium 1.4: OmniSafe pins old gymnasium via safety-gymnasium, FSRL is stale, safety-starter-agents is archived.
   - If a constraint is chosen, a small SB3 SAC subclass (cost buffer field + cost critic + λ, ≈150–300 lines) is the cheaper path. OmniSafe's `sac_lag.py` / `lagrange.py` can serve as the reference design.

---

## Sources
- Linesight repo @ f6e7822f: https://github.com/Linesight-RL/linesight/tree/f6e7822fcb91d10bafd9b8db070a1dee78046558 ; docs https://linesight-rl.github.io/linesight/build/html/index.html
- Altman 1999 (record): https://lib.ecu.edu/catalog-preview/catalog/650588
- RCPO: https://arxiv.org/abs/1805.11074 (read at https://ar5iv.labs.arxiv.org/html/1805.11074)
- PID Lagrangian: https://arxiv.org/abs/2007.03964 (https://ar5iv.labs.arxiv.org/html/2007.03964)
- Ha et al.: https://arxiv.org/abs/2002.08550
- WCSAC: https://ojs.aaai.org/index.php/AAAI/article/view/17272
- Spoor et al.: https://arxiv.org/html/2510.17564v2
- TraD-RL: https://arxiv.org/html/2603.05842
- APEX26: https://onlinelibrary.wiley.com/doi/10.1002/adrr.202500210
- Yu et al. RA-L 2025: https://pure.bit.edu.cn/en/publications/enhancing-safety-in-autonomous-racing-with-constrained-reinforcem/
- Two-stage safe RL (unopened): https://meridiancas.github.io/publications/pdffiles/2020_SMC.pdf
- FUCHS21: https://arxiv.org/html/2008.07971v2
- LEE25: https://arxiv.org/html/2504.09021
- TRI24: https://arxiv.org/html/2412.03803
- AC Gym repo @ ef0fa502: https://github.com/dasGringuen/assetto_corsa_gym/tree/ef0fa502
- WANG24: https://arxiv.org/html/2402.09075
- CAPS: https://arxiv.org/abs/2012.06644
- Christmann et al.: https://arxiv.org/html/2410.16632
- Seyde et al.: https://arxiv.org/abs/2111.02552
- Rudin et al.: https://arxiv.org/abs/2109.11978 ; legged_gym @ 8fa29acc: https://github.com/leggedrobotics/legged_gym/tree/8fa29acc6fd1910c3d9659eef6310bdd301cde0a
- Pardo et al.: https://arxiv.org/abs/1712.00378
- Knox et al.: https://arxiv.org/html/2104.13906v2
- OmniSafe @ 15603dd7: https://github.com/PKU-Alignment/omnisafe/tree/15603dd7a654a991d0a4648216b69d60b81a6366 ; PyPI https://pypi.org/project/omnisafe/
- safety-gymnasium PyPI: https://pypi.org/project/safety-gymnasium/
- FSRL @ e056fc94: https://github.com/liuzuxin/FSRL ; PyPI https://pypi.org/project/fast-safe-rl/
- safety-starter-agents (archived): https://github.com/openai/safety-starter-agents
- Earlier note: docs/research/driving_agent_sota.md (25 Oct 2026)
