# R6: top-end throttle remap (run `bcn_f1_r6`)

Status: spec + architect review, 20 Nov 2026; implemented (docs/handoff/R6_coder.md); amended 21 Nov 2026 (§2.2).
Research: `docs/research/full_throttle_sac.md` §2.1, §5 option A. Precedent: TRI24 (Subosits et al.,
arXiv:2412.03803, §III-A) maps longitudinal +0.5 to 100 % throttle so the agent can hold 100 % (paraphrase
verified in the research note).

Labels: [F] formulation / Ibrahim's decisions, [C] code or data read in this repo, [P] paper, [J] my judgement.

## 0. Decided by Ibrahim (not reopened)
- Warm start from `runs/mac/D2/bcn_f1_r5/seed0/checkpoints/step_7000000.zip` (= `models/bcn_f1_r5_step7000k`).
  Only buffer: `runs/mac/D2/bcn_f1_r5/seed0/replay_buffer.pkl` (saved at 10.35M).
- Only the throttle map changes (option A). No pedal mutual exclusion, no entropy change.
- R5 reward and R5 learning-rate schedule / hyperparameters unchanged (preset `bcn_f1_r5`).

## 1. Throttle map
### 1.1 Formula (throttle only; brake keeps the R3 deadzone map)
With `EnvConfig.throttle_remap = (k, s)` = (knee, a_sat) in action units:

    p_k = (k + 0.5) / 1.5
    pedal(a) = clip((a + 0.5) / 1.5, 0, 1)                for a <= k   (same expression as today)
             = min(1, p_k + (1 - p_k) (a - k) / (s - k))   for a >  k
    inverse(p) = released_pedal_action (-0.8)              for p == 0
               = -0.5 + 1.5 p                              for 0 < p <= p_k
               = k + (s - k) (p - p_k) / (1 - p_k)         for p > p_k   (inverse(1) = s)

**R6 values: k = 0.70, s = 0.85.** p_k = 0.8; the upper segment has slope 4/3 pedal per action unit
(2x the lower 2/3). Valid only with `pedal_deadzone=True` and `-0.5 < k < s <= 1`; else ValueError.

### 1.2 Why 0.70 / 0.85 [P: Haarnoja 2018b App. C eq. 26; derivation in research §2.1; C; J]
Small-sigma SAC equilibrium: the mean holds action a only if dQ/da >= 2 alpha a / (1 - a^2). In pedal units
(dQ/dp = dQ/da / (dp/da)):

| map | full-throttle target | needed dQ/dp |
|---|---|---|
| R5 (old) | pedal 0.95 / 0.99 / 1.0 | 19.2 alpha / 99 alpha / infinite |
| **R6 (0.70, 0.85)** | pedal 1.0 at a = 0.85 (needs dQ/da 6.13 alpha) | **4.6 alpha** |
| (0.70, 0.90) | pedal 1.0 at a = 0.90 (9.47 alpha) | 9.5 alpha |
| (0.70, 0.80) | pedal 1.0 at a = 0.80 (4.44 alpha) | 2.2 alpha, but 3x pedal noise above the knee |

Data, 7M deterministic lap 2 (`runs/mac/D2/bcn_f1_r5/seed0/laps/step_7000000_det.npz`, 2131 decisions) [C]:
43 % of decisions at a <= 0.70 (unchanged, bit-identical pedals), 20 % in (0.70, 0.85], 37 % above 0.85;
pedal >= 0.95 on 13.5 %, >= 0.99 on 0.2 %, max a 0.990. Rear slip is similar in every band (median
0.07-0.08), so the top band is not specially traction-limited.
- If the 7M actions are entropy equilibria, every state where the actor already sits at a >= 0.726 (old pedal
  >= 0.82) has dQ/dp >= 4.6 alpha, so (0.70, 0.85) lets it reach pedal 1.0 if dQ/dp stays positive to 1.0.
  With (0.70, 0.90) only a >= 0.854 qualifies. [J]
- The remap does not change the set of reachable pedals ([0, 1]) or the reward, so the max-Q pedal is unchanged;
  it only lowers the entropy price of the bound. Where full throttle hurts (wheelspin on exits), dQ/dp turns
  negative and the actor stays below s. [J]
- Above s, Q is flat in a, so the mean parks at about s and both the deterministic action and every sample above
  s give pedal 1.0.
- Step-0 shift for the unchanged 7M actor (same npz, recomputed): pedal >= 0.99 on 0.2 % -> 38.6 % of lap-2
  decisions, >= 0.95 13.5 % -> 44.4 %, mean pedal 0.631 -> 0.640. A small, intended shock.

### 1.3 Code (this repo only; ocd_first_lap untouched)
- `rlsetup/driving/config.py` `EnvConfig.throttle_remap: tuple | None = None`. None = today's map.
  `to_dict` omits it while None, so every existing config.json, demo key, corner/QSS key and env hash is unchanged.
  Set -> written to config.json as `env.throttle_remap: [0.7, 0.85]` (the AC side reads it later).
- `rlsetup/driving/demo_driver.py`: one pair of functions, `throttle_pedal(a, remap)` / `throttle_action(p, remap,
  released)`. `action_to_inputs(..., throttle_remap=None)`, `pedal_to_action` / `inputs_to_action` gain the same
  kwarg for the throttle only. remap None must take today's code path literally (no refactor of the expression).
- `rlsetup/driving/env.py` `to_inputs` passes `cfg.throttle_remap`; the settle `DemoDriver` gets it too, so settle
  pedals (and hand-over states) are the same as under R5, up to float32 rounding of the action.
- `rlsetup/driving/evaluate.py` `_pedals(actions, pedal_deadzone, throttle_remap=None)`; `lap_r4` and every caller
  pass the run's remap (otherwise the full-throttle metric reads a = 0.85 as pedal 0.9).
- `rlsetup/driving/bc.py` `pedal(a, pedal_deadzone, throttle_remap=None)` for the throttle column (BC is unused in
  R6, but keep it consistent).
- Refuse `throttle_remap` with `reward.mode == 'ref_tracking'`: the legacy smoothness term du is on raw actions,
  so under reward (1) the remap would change the reward. Under time_pbrs smoothness is logged only [C: env.py
  step, `r = progress - time - penalty`, line, exit]; note in the env docstring that the legacy `cost` /
  `smoothness` of R6 laps are not comparable with R5's (`cost_r1` = time is).
- **AC guard (required now, one check):** `rlsetup/assetto/actor.py` `_config` raises
  `ValueError('unsupported source env.throttle_remap')` if `env.get('throttle_remap') is not None`. Today it checks
  only listed keys, so an R6 config would pass and `assetto/driving.py controls` (l. 590) and `SettleDriver.act`
  (l. 633) would silently use the old map. Full AC support (controls + settle inverse + `pedal_formula` string in
  actor.py l. 170, `scripts/ac_preflight.py`, `scripts/comparison/*` labels) is a **follow-up stage**.

## 2. Replay buffer and critic
### 2.1 Relabel (exact) [C; P: PEBBLE, Lee et al. 2021, relabel stored experience when its meaning changes]
For every stored throttle value v produced under the old map: `v_new = throttle_action(pedal_old(v), remap_new)`
(identity for v <= 0.70, bitwise; v -> 0.70 + 0.75 (pedal_old(v) - 0.8) above; v = 1.0 -> 0.85).
Apply it to:
- `actions[..., 1]`;
- the throttle slots of `observations` and `next_observations`: the observation holds the last action and the
  (history - 1) previous actions [C: env.py `_decision_observation`, `_push_history`]. For bcn_f1_r5 (obs 159 =
  9 ego + 3 + 120 preview + 3 last action + 2 x (9 + 3)) the slots are **133, 145, 157**. Compute them with a
  helper `env.throttle_obs_indices(cfg)` (also handles obs_gear), never hard-coded. The "nothing pressed" -1 and
  settle-phase actions are old-map actions too, so the same function applies.
Everything else is untouched: rewards, dones, timeouts, corner_bonus, crash_near, warm-up mask, steering, brake.
The executed pedal, the physics, the reward and the next state of every transition are unchanged, so the
relabelled buffer is the buffer an R6-map driver with the same pedals would have written (to float32 rounding:
|pedal_new(v_new) - pedal_old(v)| <= 2e-7). Dropping the rows is rejected: 57 % of decisions are above the knee.
New `warm.throttle_relabel: bool = False` (omitted while False) runs it in `agent.preload_buffer`, from the
source config's map (absent key = legacy) to `cfg.env.throttle_remap`. Stats to status.json /
provenance.json `init_buffer.throttle_relabel`: rows and obs cells changed, max |delta pedal|; abort
(`init_mismatch`) if that max exceeds 1e-6.
`warm_start` compatibility: `throttle_remap` may differ from the source only when `warm.throttle_relabel` is on;
every other env field must still match.
- Off-policy note: the buffer holds 9.35M-10.35M transitions, from a later policy than the 7M actor and critic.
  That is fine for SAC (off-policy). All of it was collected at the lr floor (scale 0.01 from 6.2M), so the
  behaviour is close to 7M (lap 2 85.27 s at 7M, 85.74 s at 10.35M) [C: evals.jsonl].
- The relabelled buffer has no throttle in (0.85, 1]. That is exact (the preimage of pedal 1.0 is [0.85, 1]),
  but it leaves the critic to extrapolate there until online data arrives (section 2.2).

### 2.2 Critic phases (frozen actor) [C: train.py docstring, init_mode 'warm']

> **Amendment, 21 Nov 2026 (Ibrahim): offline warm-up removed.** The `bcn_f1_r6_smoke` gate failed in the offline
> phase at 3e-5: gate 0.0006 before training -> 0.0032, 0.0066, 0.0100, 0.0137, 0.0193 over 5k grad steps (critic
> loss 52.5 -> 77.9, mean Q 220.2 -> 211.7). Judged as lr drift of the converged critic refit on a fixed buffer, not
> damage from the relabel (the loaded critic already scored 0.06 % on the relabelled rows). Now
> `offline_critic_grad_steps = 0` in `bcn_f1_r6` and `bcn_f1_r6_smoke` with no offline lr override. The gate is kept
> as a one-shot pre-training check on the loaded critic after the relabel (tol 1 %; warmup.jsonl row at grad_steps 0).
> learn() then starts with the online critic phase at R5's lifetime lr (scale 0.01: critic 3e-6), actor and alpha
> frozen for 100k steps (20k in the smoke), retries and crash factor 3 from step 0. Item 1 below no longer runs.
1. **Offline critic: 50,000 grad steps** on the relabelled buffer (blocks of 1,000), critic lr **3e-5**
   (scale 0.1, this phase only; R5's floor 3e-6 is too slow to refit a re-parameterised input slice, my
   judgement; the gate below decides). New gate in warmup.jsonl: on a fixed batch of 4,096 relabelled rows
   (old throttle > 0.70; s_new = relabelled obs), `mean |Q_min_new(s_new, a_new) - Q_min_src(s_old, a_old)| / mean |Q_min_src|` with Q_src the
   loaded 7M critic on the original row. **Pass <= 1 %** and not rising over the last 5 blocks. Fail -> stop
   before learn() and report (do not launch).
2. **Online critic: 100,000 env steps** (`warm.critic_warmup_steps = 100_000`, ~7 min at 250 steps/s), actor and
   alpha frozen, the stochastic 7M actor acting under the new map. It fills (0.85, 1] with real rows (37 % of
   deterministic decisions sit there) before the actor sees critic gradients there. lr scale 0.01.
3. Unfreeze at 100k: actor lr 1e-4 x 0.01 = 1e-6, critic 3e-4 x 0.01 = 3e-6, as R5 at 7M. Critic check at the
   first eval after unfreeze (`critic_check_tol` 0.08, unchanged). CollapseGuard: baseline = probe of the loaded
   actor under the new map, threshold 0.4 (R5).

### 2.3 "Schedules unchanged" needs the lifetime step axis [C]
Warm mode restarts `num_timesteps` at 0, and LrDecayCallback, RetryCallback and CrashEmphasisCallback read
`num_timesteps`. Taken literally, preset `bcn_f1_r5` in a warm run would put the lrs back at 1.0 x (100x what
the 7M actor trained at, until 3M), and switch retries and crash emphasis off for 500k. That would be three
changes, not one. Fix: `TrainConfig.lifetime_schedules: bool = False` (omitted while False); on, these three
callbacks use `num_timesteps + lifetime_steps_offset` (7,000,000 here). Result: lr scale 0.01 from step 0,
retries and crash emphasis on from step 0, exactly as R5 at 7M. Do not change `env.retry.start_step` instead:
it is in the env block and would trip the warm-start env check.

## 3. Preset, run, metrics
- Preset `bcn_f1_r6` = `_bcn_f1_r5()` with: `agent.init_mode='warm'`; default init_from / init_buffer = the two
  paths in section 0; `env.throttle_remap=(0.70, 0.85)`; `warm`: relabel False, speed_relabel False,
  throttle_relabel True, offline_critic_grad_steps 50,000, offline critic lr scale 0.1, critic_warmup_steps
  100,000 (actor_lr 1e-4 inherited); `lifetime_schedules=True`; `total_timesteps=4_000_000`, plateau stop off.
  Reward, corner bonus (never fade -> laps aided, best_aided.zip), apex split, line, setup `bcn_f1_best`, IQP w3.0,
  alpha (auto_0.1, target -3), gamma, n_steps 3, batch, UTD 0.5, 4 envs on the Mac: unchanged.
- Run dir `runs/mac/D2/bcn_f1_r6/seed0`, Mac keeper as R5. Commit before launch (strict provenance).
- Ibrahim judges at 0 (step-0 eval), 0.5M, 1M, 2M, 3M, 4M.
- Metrics per eval (R5 references from evals.jsonl):
  - deterministic lap 2 vs R5: 85.27 s at 7.0M (85.23-85.74 s across 6.9M-10.35M), best 84.87 s at 4.95M; within
    +-0.3 s is inconclusive (DECISIONS #27 rule);
  - deterministic clean rate, stochastic clean rate over 10 laps (R5 0.9-1.0), stochastic mean lap (87.6 s at 7M);
  - per straight S_k (`lap_r4` straights): throttle-pedal share >= 0.99 (new key `full_throttle_share_99`) and
    >= 0.95, brake share, top speed;
  - per straight, new `time_s`, `qss_time_s` (integral of ds / v_qss over the same span, corner-table speed) and
    `time_lost_s`; their sum vs the ~3.6 s of the research note. Lap reference QSS 80.1 s;
  - throttle action share in (0.85, 1], alpha, policy entropy, Q vs MC, crash causes per unit.
- Pre-launch (sim-runner, no training): (a) the new straight metrics on R5's 7M lap under the old map (the R6
  baseline); (b) a deterministic 2-lap eval plus 10 stochastic laps of the unchanged 7M actor under the new map.
  If (b) crashes deterministically, report before launching; it is information, not a stop.
- Stop and escalate [J]: 500k steps after the unfreeze, best clean det lap 2 > 85.9 s and stochastic clean rate
  < 0.7 over the last 5 evals.

## 4. Tests (rl-coder)
1. Off = bit-exact: for remap None, `action_to_inputs`, `pedal_to_action`, `inputs_to_action`, `evaluate._pedals`,
   `bc.pedal` equal HEAD on a grid incl. -1, -0.5, 0.7, 0.85, 1 and random float32; a 200-decision env stream
   hash of the `bcn_f1_r5` env equals HEAD; `to_dict()` of every preset in PRESETS has no `throttle_remap`; demo
   dataset key unchanged.
2. Map: continuous at k (both sides), pedal(s) = 1 = pedal(1), non-decreasing on [-1, 1]; bitwise equal to the old
   map for a <= k; p -> a -> p round trip <= 1e-12 (float64) and <= 2e-7 through float32 for p in [0, 1];
   a -> p -> a on (-0.5, s]; pedal 0 -> released action; invalid (k, s) and remap with pedal_deadzone False or
   ref_tracking raise. Brake identical to the old map.
3. Relabel: pure function vs old pedal (<= 2e-7, 10^6 random float32 + edges 0.7, nextafter(0.7), 0.985, 1.0);
   `throttle_obs_indices` against a real env observation (step a distinct throttle, read the slots; obs_gear on
   and off); **real transitions**: roll the bcn_f1_r5 env 100 decisions under the old map with a fixed seed,
   relabel, replay the relabelled actions under the new map from the same reset: obs, rewards, next obs match the
   relabelled records to 1e-5, and the replayed obs throttle slots equal the relabelled ones; a synthetic
   BonusNStepReplayBuffer (bonus, crash_near, warm-up set) changes only actions[..., 1] and the obs slots.
4. Config / warm start: the `bcn_f1_r6` config.json holds `env.throttle_remap: [0.7, 0.85]` and the provenance
   relabel stats; warm start from a legacy-map source refuses without `throttle_relabel` and passes with it;
   AC `_config` refuses a config with `throttle_remap`.
5. Lifetime schedules: offset 7M -> lr scale 0.01 at step 0 (also in the offline phase, where it is the override
   0.1), retries and crash emphasis active at step 0; off -> unchanged.
6. Eval: `_pedals` with the remap; `full_throttle_share_99`; straight `time_lost_s` on a synthetic trace.
Full suite green except the known failures; a 150k smoke (`bcn_f1_r6_smoke`: offline 5k, online 20k) before launch.

---

## Architect review (rl-architect, 20 Nov 2026)

**Verdict: APPROVE**, for the spec as written above (the items below are already in it). No OPEN item (§11)
or superseded item (§12) is touched: actions stay 3 absolute outputs, the driver still does not see the setup,
and reward (1)/time_pbrs and the lap cost are unchanged.

| Layer | Verdict |
|---|---|
| Action map (0.70, 0.85) | APPROVE. TRI24 precedent [P]; needed dQ/dp falls 19-99 alpha -> 4.6 alpha; 43 % of decisions bit-identical [C] |
| Buffer relabel | APPROVE. Exact up to float32; it must also cover the obs action slots (133/145/157), not only `actions` [C] |
| Critic warm-up | APPROVE-WITH-NOTES. Offline at lr 3e-5 is my call outside the R5 RL schedule; the 1 % gate decides; online 100k fills (0.85, 1] |
| Schedules | APPROVE only with `lifetime_schedules`. Without it the "unchanged" preset gives a 100x lr and 500k without retries or crash emphasis [C] |
| Eval metrics | APPROVE. `_pedals` must use the remap, or the headline metric is wrong |
| AC | One refusal check now; full support is a follow-up |

Reasons this is downstream-safe: the executed pedal range is still [0, 1] and the reward is untouched, so lap time
(`cost_r1`) still ranks setups only by time. The remap removes an entropy artefact that would otherwise cost every
setup a similar ~straight-line deficit, and it is not tied to the setup. For fine-tunes it is a fixed
parameterisation, applied the same way to every setup.

Risks (metric):
- Exit wheelspin from the 2x pedal gain above the knee: stochastic clean rate, crash causes and rear slip p90 on
  exits per unit vs R5.
- Critic extrapolation in (0.85, 1] just after the unfreeze: Q vs MC at the first evals; the share of throttle
  actions above 0.85 jumping to ~1 with no lap gain.
- The lr floor (1e-6 actor) is too slow to use the new headroom: straight `time_lost_s` flat after 1M while
  `full_throttle_share_99` rises. Ibrahim's decision stands; this is only flagged.
- Brake dabs on straights are not addressed (option B is excluded): brake share per straight; expect no change.
- Alpha drifts because the bound now costs less entropy: alpha and policy entropy vs R5's 0.063 / -2.92.

Questions for Ibrahim: none blocking. The offline-phase critic lr (3e-5, section 2.2) is the only setting outside
the R5 values; veto it if "unchanged" was meant to cover the warm-up phase too, and the gate then runs at 3e-6.
