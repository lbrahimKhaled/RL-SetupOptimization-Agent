# R2 — clean-corner exit bonus (faded), focused starts, low-noise rollouts

Status: DRAFT by lead, 7 Nov 2026. Needs rl-architect review (fills every [ARCH] item) before rl-coder starts.
Research basis: `docs/research/plateau_fixes.md` (A.3, A.5, B, D-F). Decisions: `docs/DECISIONS.md` #22.

## 0. Why
`f1_best_w35` (sf70h_f1, setup sha_f1_best, ξref = IQP w3.5, time_pbrs) plateaued: best clean deterministic
lap 2 102.42 s at 2.5M → 101.87 s at 9.5M; deterministic eval clean in 50/248 evals; stochastic ~60 % clean,
mean ~104.5 s. Off-tracks cluster at s ≈ 2350–2450 m (46), 750–850 m (28), 1550, 5100, 650, 4700 m.

Step-1 diagnostic (lead, 7 Nov; script in the session scratchpad, numbers approximate because the RL position
is mapped to the w3.5 line by nearest point): QSS on the w3.5 line = 93.83 s (safe line 100.16 s).
- In the last 150 m before a crash the driver peaks at ~1.2 × v_qss; clean laps do the same in those zones.
- At most corner apexes his minimum speed is ≥ v_qss (e.g. 814 m: 27.3 vs 24.1 m/s; 3113 m: 23.7 vs 19.4).
- Time is lost mainly on exits: apex → apex+150 m costs +0.2 to +0.57 s per corner vs QSS
  (1586 m +0.52, 5129 m +0.57, 3113/3167 m +0.35); all corner windows together +4.46 s.
So: too fast in, too slow out. A bonus on entry/apex speed capped at v_qss would give ~zero signal.

## 1. Decisions by Ibrahim (7 Nov 2026) — not to be changed by the team
1. **Clean-corner exit bonus**: for each corner, paid once at the corner's exit point, only if the car stayed
   on track through the whole corner window (no off-track / wrong-way / non-finite between window start and
   the exit point). Size grows with exit speed relative to the QSS exit speed, **capped at v_qss** (no extra
   for exceeding it). Fast-and-crashed earns nothing.
2. **Training aid only**: bonus weight w(t) = 1 → linear fade → 0. After it reaches 0, **≥ 4M env steps with no
   bonus** before any lap time from this run is reported as the result. The final driver is rewarded on
   time_pbrs only (DECISIONS #2 restored).
3. **One run** containing all changes: exit bonus + focused starts (research option B) + low-noise rollouts
   (option F). No control run.
4. **Warm start** from `runs/mac/D2/f1_best_w35/seed0/checkpoints/best.zip` (101.87 s, ~9.5M steps) on car
   `sf70h_f1`, setup `sha_f1_best`, ξref `csv:docs/results/shanghai_line_setup/line_iqp_w3.5.csv`, Shanghai.
5. **Stop**: keep training until Ibrahim stops it, or a clean deterministic lap 2 **≤ 95 s** (keeper rule).
6. TensorBoard: the new run next to `f1_best_w35` (best sf70h_f1 Shanghai run).
Not in scope: TQC / lower late lr (option E), network resets (option C).

## 2. Exit bonus — requirements and [ARCH] items
- **QSS reference for the bonus**: v_qss on the **w3.5 line** (`qss_lap.lap_time(path=line, car='sf70h_f1',
  ggv_data=<cached sha_f1_best profile's ggv/traction>)` returns `v_profile`), mapped to Track.path arc length
  (the env's progress coordinate). The existing Track.path profile stays for start speeds. Cache it with a key
  like `qss_profile.cache_key`.
- **Corners**: detected once from the w3.5 QSS profile (local minima of v_qss below a threshold; the diagnostic
  found 13 apexes at 614, 740, 814, 878, 1586, 2094, 2329, 2531, 2629, 3113, 3167, 3325, 5129 m on Track.path).
  [ARCH] detection rule; merging of close apexes (814/878, 3113/3167); window start (e.g. braking point or
  apex − X m); **exit point** (apex + d, or where v_qss recovers to a fraction of the next straight's speed,
  or the start of the next braking zone) — the diagnostic used apex + 150 m.
- **Size**: bonus_k = B · w(t) · f(v_exit / v_qss_exit), f increasing, f(≥1) = f(1) (cap), f small or 0 below
  a floor ratio. [ARCH] B, f, floor. Constraints from research A.3: a crash costs ≈ P + V ≈ 360 return units;
  per-lap bonus total must stay small next to the lap return (~ a·5450 − b·100); B such that it can't justify
  a visible increase in crash rate. Exit speed measured as a single sample at the exit point or averaged over a
  short window [ARCH].
- **Replay buffer**: store the bonus as a separate per-transition field (unweighted), and apply w(t) at sample
  time so the fade also applies to old transitions; must work with n-step 3 (SB3 builds n-step returns at
  sample time — see `rlsetup/driving/relabel.py` docstring). Warm-start buffer (1M transitions from
  f1_best_w35, no bonus field): [ARCH] bonus = 0 for those, or recompute from stored info, or start with a
  fresh buffer.
- **Schedule** [ARCH, within Ibrahim's frame]: research suggests on ~1.5–2M steps, fade ~2M (linear or
  clean-rate-gated), then ≥ 4M bonus-free (Ibrahim). The current w(t) must be logged and saved in checkpoints
  so a resume continues the schedule.
- **Evaluation** unchanged: deterministic + stochastic laps, lap time is the measure; eval returns reported
  without bonus. Log per corner: exit speed ratio, bonus paid, clean-corner counts; per-lap bonus sum; w(t).

## 3. Focused starts (research B) — requirements and [ARCH] items
- 20–30 % of resets start 100–300 m before a hot spot; ≥ 70 % stay uniform (current D1 rule). [ARCH] fraction.
- Hot spots adaptive: 50 m bins on Track.path weighted by recent training off-track counts (window [ARCH]),
  seeded from f1_best_w35's crash bins at launch; floor weight so a fixed corner is not forgotten.
- Start state: the existing settle mechanism (on ξref, v0 = f · min v_qss over the window) unless restoring
  visited states is feasible with OCDBackend [ARCH — check; do not edit ocd_first_lap].
- Log: share of focused starts, current bin weights, clean rate per bin.

## 4. Low-noise rollouts (research F) — requirements and [ARCH] items
- Some training rollouts act with reduced or no SAC sampling noise so the buffer contains the fast-but-fragile
  deterministic behaviour that evaluation sees. [ARCH] which of the 4 envs / what fraction of episodes, noise
  level (mean action vs scaled std), interaction with the entropy target and the guard.

## 5. Preset, run, tooling
- Preset `f1_r2` (run name `f1_r2`): f1_best_w35 + the above; `init_from` best.zip; warm-start procedure as
  `r1_warm` where it applies [ARCH: critic warm-up needed? Reward changes, so yes probably].
- Keeper: `runs/mac/keeper_f1.sh f1_r2` with the ≤ 95 s rule (current script uses < 95; make it ≤ 95).
- TensorBoard: `--logdir_spec` with the new run and `runs/mac/D2/f1_best_w35/seed0`.
- Provenance: the bonus QSS cache key, corner table, schedule and focused-start config in config.json /
  provenance.json.

## 6. Acceptance (before launch)
- Tests: bonus = 0 when the corner window has a failure; capped at v_qss; paid exactly once per corner pass;
  w(t) schedule and resume; buffer weighting reproduces stored reward when w = 0 (bit-identical to time_pbrs);
  focused-start fraction within tolerance over N resets; low-noise envs act with the configured noise.
- Full test suite green; sim-runner smoke (≤ 30 min) shows bonus values in the expected range and no change
  in progress/time terms; ocd_first_lap untouched.
