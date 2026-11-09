# C1 — Setup-critic warm start from F1 2020 setups

Status: DRAFT by lead, 9 Nov 2026. Needs rl-architect review ([ARCH] items), then rl-coder.
Scope: the critic of the TD3 setup agent only. The TD3 actor, its training loop and the OCD fine-tune are NOT part of C1.

## 0. Why (measured in the session scratchpad, ridge models, read-only)
- Data: `f1_2020_all_team_setups.csv` (2,565 rows, 196 drivers after cleaning, 22 tracks; f1laps.com, F1 2020).
- Session + controller + team: held-out R² 0.320. Adding driver id: 0.643. Setup knobs add +0.034 on top of driver.
- Same driver + track + session, different setups (238 groups, 593 laps): setup predicts R² 0.110 of the lap change on unseen
  driver/track groups; mean change 0.94 % (0.85 s on a 90 s lap).
- Earlier file, without driver id: correlation clusters, game-menu groups and physics groups were indistinguishable from random
  partitions (0.482–0.487 vs random 0.482 ± 0.006). Knob correlations mostly reflect copied setups (one setup appears 315 times).
- Critics trained separately and summed: 0.212; jointly: 0.497 (physics groups, ridge). Summing double-counts correlated knobs.
- Limits to state in every result: labels come from a game car, not the OCD car; setup signal is small; this is a warm start, not a J(p) model.

## 1. Decisions by Ibrahim (9 Nov 2026) — not to be changed by the team
1. Purpose: a **warm start** for the TD3 critic that should be decent; it is later fine-tuned on OCD `(p, J)` runs.
2. **Critic inputs are the 15 OCD knobs the data covers.** The other 11 OCD knobs (dampers 4, heave 3, packers 2, final drive, diff preload) are dropped; they are not inputs.
3. Brake pressure and the four tyre pressures are dropped (no OCD knob).
4. **Groups come from PCA first.** If PCA groups perform badly, drop them and use the physics groups of §4.
5. **Track is abstracted** into a few summary numbers (corners, longest straight, ...). No raw track points or racing line.
6. Setup agent is **TD3**. Design the critic for TD3; do not design the TD3 itself.
7. **Few inputs, because the dataset is small.** Do not feed drivers/teams as network inputs.
8. **Frequency as mediator:** no knob is dropped, the search space shrinks. Springs are expressed as ride frequencies and the second knob (roll stiffness / share) is derived through them, so impossible spring/bar combinations cannot occur. Follow Assetto Corsa if it has an equivalent; otherwise use §3. Checked (`docs/research/ac_spring_frequency.md`, community sources, no Kunos specification found): AC exposes per-corner spring rates and per-axle bars as independent knobs, frequency is at most a read-out, and no coupling is documented. AC has no equivalent, so §3 applies.

## 2. Dataset (`rlsetup/critic/data.py`)
CSV path is a CLI argument; the file stays out of the repo (contains public usernames). Record its sha256 in provenance. Driver names map to integer indices at load; no username is ever written to an output.

Pipeline, in order, with row counts logged at each step:
1. Keep conditions = Dry, all 15 knobs present.
2. `gap` = 100 · (lap / median lap of the track − 1), median over rows surviving step 1.
3. Drop `gap > 8 %` (slow-lap outliers; the earlier file's slowest dry lap was 5.9 %). Then recompute the median and `gap` once.
4. Keep the 17 tracks with a TUM centerline (table below). Drop Azerbaijan, France, Monaco, Singapore, Vietnam.
5. Normalise session to {Race, Quali, TT}; controller to {Wheel, Controller, blank} (blank is its own category).

Expected: about 2,007 rows. Nuisance (linear, §5): session, controller, team, driver (drivers with < 3 rows pool into `rare`).

| F1 2020 | TUM file | F1 2020 | TUM file |
|---|---|---|---|
| Abu Dhabi | YasMarina | Hungary | Budapest |
| Australia | Melbourne | Italy | Monza |
| Austria | Spielberg | Japan | Suzuka |
| Bahrain | Sakhir | Mexico | MexicoCity |
| Belgium | Spa | Netherlands | Zandvoort |
| Brazil | SaoPaulo | Russia | Sochi |
| Canada | Montreal | Spain | Catalunya |
| China | Shanghai | United States | Austin |
| Great Britain | Silverstone | | |

## 3. Game knobs → 15 OCD knobs (`rlsetup/critic/knobs.py`)
Inputs to the network are the OCD knobs normalised to [−1, 1] with the `low`/`high` of `ocd_lap.car_setup.KNOB`, in this order:
`ride_freq_front_hz, ride_freq_rear_hz, roll_stiffness_Nm_per_rad, roll_stiffness_front_share, ride_height_front_m, rake_m, aero_level, aero_front_share, brake_bias_front, camber_front_deg, camber_rear_deg, toe_front_deg, toe_rear_deg, lsd_lock_drive, lsd_lock_coast`.

| OCD knob | From game | Status |
|---|---|---|
| camber F/R | degrees, direct | direct |
| toe F | + game value (toe-out = +) | [ARCH] sign assumed |
| toe R | − game value (game rear toe treated as toe-in; OCD toe-out = +) | [ARCH] sign assumed |
| brake bias | pct / 100 | direct |
| lock drive / coast | (pct − 50) / 50 | assumed linear |
| ride_freq F/R | spring click c: 2.5 + 5.5 (c − 1) / 10 Hz | assumed linear |
| h_F | 20 + 6 (click − 1) mm | assumed |
| rake | h_R − h_F, h_R = 40 + 8 (click − 1) mm | assumed |
| aero_level | (fw + rw − 2) / 20 | assumed |
| aero_front_share | 0.35 + 0.25 (fw/(fw+rw) − 1/12) / (10/12) | assumed |
| K_φ, χ | through frequency, below | mediator |

**Frequency mediator.** With m_c = 171 / 199 kg (front/rear), t = 1.6 / 1.5 m:
K_w = m_c (2π f)²; bar click c gives K_b = 21 000 + 16 900 (c − 1) N·m/rad (21–190 kN·m/rad);
front roll = K_b,F + ½ K_w,F t_F²; rear roll likewise; K_φ = front + rear; χ = front / K_φ.
This inverts the repo's K_b = χ K_φ − ½ K_w t² ≥ 0, so K_b ≥ 0 by construction.

Rows whose derived knobs fall outside the OCD bounds (χ, K_φ, rake) are combinations OCD cannot represent: **drop them**, log the count per knob. If more than 5 % of rows go, stop and report.

Click-to-physical maps are approximations (the game publishes none); the OCD fine-tune corrects them. All constants live in one config so they can change later.

Known caveat to keep in results: `lsd_lock_coast` has no effect in OCD until engine braking exists.

## 4. Groups (physics fallback and PCA)
Physics (5 groups): aero + heights {aero_level, aero_front_share, ride_height_front, rake}; springs + roll {ride_freq F/R, K_φ, χ}; alignment {camber F/R, toe F/R}; diff {lock drive, coast}; brakes {bias}.

PCA grouping, fitted **inside each training fold only**: deduplicate on (track, setup), centre knobs by track median, standardise, PCA, keep k components by parallel analysis (95th percentile, 200 shuffles, k ≥ 2), varimax rotate, assign each knob to the component with the largest |loading|.

Controls: one group holding all 15 knobs; 20 random partitions with the PCA group sizes.

## 5. Model (`rlsetup/critic/model.py`, torch)
- Track features z (5, `rlsetup/critic/track_features.py`, TUM centerline, smoothed curvature, thresholds from PDF Table VIII): `n_corners, longest_straight_m, straight_share, slow_corner_share, fast_corner_share`; standardised on training tracks. [ARCH] smoothing window and corner merge rule.
- `GroupCritic(groups, z_dim)`: Ĵ(p, z) = b + Σ_g h_g(p_g, z). Each h_g is an MLP (hidden 32, 32; **SiLU, not ReLU**, so ∂Q/∂p is smooth for the TD3 actor). [ARCH] width/depth.
- Output unit: % lap-time gap, **lower is better**. `q(p, z) = −Ĵ` keeps TD3's maximise convention.
- Nuisance offset, game data only: linear with L2, fitted jointly with the heads; used in training, **never** in `q`.
- `TwinCritic`: two independent `GroupCritic`s (different seeds and bootstrap resamples). `q1, q2`, and `q_min = min(q1, q2)` (equals max Ĵ, pessimistic for a cost). γ = 0 bandit, so no target networks.
- Interface reserved for later: z_car, z_drv absent in C1; `forward(p, z)` must stay simple. [ARCH] whether p0 belongs in the critic state (proposal: no; J depends on p, not on where search started).

## 6. Training (`rlsetup/critic/train.py`)
MSE on `gap`, Adam, weight decay, batch 256, early stop on an inner 15 % validation set split by setup hash, CPU, `torch.set_num_threads(1)` (f1_r4 is running on this machine). Modes:
- `joint`: all heads and the nuisance offset, one loss.
- `separate`: Ibrahim's original idea. Fit nuisance, then each head alone on what nuisance leaves.
- `backfit`: each head in turn on the residual of nuisance and the other heads, 5 cycles.

## 7. Evaluation (`scripts/critic_warmstart.py`, report to `docs/results/critic_c1.md`)
Every fitted thing (nuisance, PCA, scalers, track scaling) is fit on training folds only.
- **Held-out setups:** 5-fold × 3 repeats, grouped by setup hash. R² gain over nuisance-only.
- **Within-driver ranking:** folds grouped by (driver, track). In same driver + track + session groups with ≥ 2 different setups, pairwise order accuracy and Spearman of the setup-only prediction. This is the headline number.
- **Held-out track:** leave one track out over the 17 tracks, same ranking metric.
- Baselines: nuisance only; ridge with 15 main effects (the number to beat).
- Runs: {PCA, physics, one group} × {joint, separate, backfit}; random partitions in `joint` only.

**Decision rule, fixed before running:** keep PCA groups only if, in `joint`, their within-driver pair accuracy and held-out-setup R² gain are both ≥ physics groups **and** both above the 95th percentile of random partitions. Otherwise use physics groups. Report honestly if physics also fails to beat random.

**Warm-start bar:** in `joint`, pair accuracy > 0.5 with the 95 % CI excluding 0.5 and at least equal to the ridge baseline. [ARCH] set the final bar. If not met, report it; do not tune until it passes.

## 8. Tests and outputs
`tests/test_critic_*.py`: row counts on a small fixture; knob map and inverse formula (K_b ≥ 0, known values); PCA fit does not see held-out rows; determinism for a fixed seed; twin gives independent outputs; `q` ignores nuisance; gradient of `q` wrt p is non-zero and finite. Trained weights go to `runs/critic/` (gitignored) with `run_provenance()` and the CSV sha256. No new dependencies.
