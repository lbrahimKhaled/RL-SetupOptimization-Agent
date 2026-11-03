# f1_sha: SAC driver on Shanghai with car sf70h_f1 (status 3 Nov 2026, 14:02)

Run `runs/mac/D2/f1_sha/seed0`, preset `f1_sha`, rlsetup commit `3caf05f` (branch sf70h-f1, pushed).
ocd_first_lap = worktree `/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1` (branch sf70h-f1,
commit `ee86683`, clean). It is imported through `PYTHONPATH`. The venv `.pth` and the main checkout
are unchanged. provenance.json records `ocd_first_lap.path` = the worktree, commit `ee86683…`, dirty False.

## What changed (commit 3caf05f)
1. **Steering scale.** Steering action +-1 now maps to +-the car's lock (`demo_driver.steer_lock_rad`):
   `CarSpec.steer_max_rad` if the CarSpec has it, else 0.6 rad. sf70h_f1 gives 0.30 rad. OCD's
   actuator clips at the same value. `DrivingEnv.to_inputs`, `action_to_inputs` / `inputs_to_action`
   and the demo/settle pure pursuit all use it. av21 and sf70h_tum keep 0.6. Their demo streams
   (presets full, p1_TUM) are bit-identical before and after the change (sha of obs, actions and
   rewards over 2 episodes).
   **Observation scale `ObsConfig.steer_scale_rad` kept at 0.6**: the same rad-to-input scale for
   every car, and the observation layout stays frozen. For sf70h_f1 the actual-steer input lies in
   [-0.5, 0.5] = action / 2, so it is no longer in action units. Nothing breaks.
2. **Demo/settle traction control (f1_sha only).** `traction_slip_start` 0.06, `traction_slip_span`
   0.03. Throttle is cut fully at rear slip 0.09, just past the new Fx peak (kappa 0.075-0.081).
   PLACEHOLDER. Other presets keep 0.08 / 0.12.
3. **Setup `p1_TUM_f1`** (`setups.CAR_SETUPS`, built on demand). It is p1_TUM's knob changes on
   SF70HF1Setup (`for_car`), so heave is on at 12.95 / 5.85 Hz. **Brake bias = the car's 0.58,
   not p1_TUM's 0.49.** Reason: 0.58 is the car's sourced default, and the ~102.4 s QSS reference
   in SF70H_F1_UPGRADE.md §3 uses it ("bias 0.49" is listed there as a separate variant).
   `heave_damping_ratio` = 0.3 (PLACEHOLDER, no F1 source).
4. **Preset `f1_sha`** = p1_TUM with car sf70h_f1, setup p1_TUM_f1 and the TC above. That means:
   time_pbrs from scratch, demos f in (0.5, 0.8), BC gate not required, ERS off, 15 kN·m, safe
   line, no step or wall cap. The demo gate stays at 0.8; it passed, so 0.75 was not needed.
5. QSS profile `data/qss_profile_Shanghai_p1_TUM_f1_sf70h_f1_brake15000.npz`: **102.53 s**. This
   equals the report's "sf70h_f1 + heave damping 0.3" value.
6. Tests: `tests/test_f1_sha.py` (9 tests; the sf70h_f1 ones skip without the worktree).

## Smoke checks (before launch)
- Demos (40 episodes, noise as preset): clean 0.85 with the new TC vs 0.775 with the old 0.08/0.12.
- Full demo collection (140 episodes, 200,672 transitions): **clean 0.92**, gate passed.
- Noiseless demo flying laps (lap 2): f 0.6 172.1 s, f 0.7 149.2 s. f 0.8 leaves the track at
  s ≈ 1.4 km: the car weaves laterally (e ±1 m) at 63 m/s.
- Fast suite `.venv/bin/python -m pytest -q -m "not slow"`:
  - main checkout: 290 passed, 5 skipped, 2 failed.
  - worktree (`PYTHONPATH=`): 291 passed, 4 skipped, 2 failed.
  - The 2 failures are the known ones: provenance cars.py and the qss cache key.

## Commands
```
cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent
caffeinate -dimsu nohup env PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 \
  platforms/mac/train.sh --preset f1_sha --seed 0 --run-name f1_sha > runs/mac/f1_sha_stdout.log 2>&1 &
nohup runs/mac/keeper_f1.sh f1_sha > runs/mac/keeper_f1_sha.log 2>&1 &
.venv/bin/tensorboard --logdir_spec "f1_sha:runs/mac/D2/f1_sha/seed0,p1_TUM:runs/mac/D2/p1_TUM/seed0" \
  --port 6006 --host 127.0.0.1 --reload_interval 30
```
The keeper is `runs/mac/keeper_f1.sh`: a copy of keeper.sh that also exports PYTHONPATH=worktree,
so resumes import the f1 car. It stops at the first clean deterministic lap 2 < 95 s and resumes
at most 5 times.

PIDs (launched 13:44):
- training python: 89599
- caffeinate wrapper: 89601
- keeper: 89920
- TensorBoard: 89933 (http://127.0.0.1:6006)

Stop: `pkill -f "keeper_f1.sh f1_sha"; pkill -INT -f "run-name f1_sha"`

## First evaluations
- Demos 0.92 clean. BC gate failed (off track; recorded, not required). p1_TUM's BC gate failed
  the same way.
- Unfreeze at 20k. One guard rollback at 40k (probe 447 m vs baseline 919 m). p1_TUM, bcn and
  p0_TUMref each had exactly one rollback at 30-40k. The guard window closes at 220k with no
  second collapse so far.

| step | det lap | stochastic clean | sto mean lap | alpha | env steps/s |
|---|---|---|---|---|---|
| 0 | off track (1013 dec.) | 0/10 | – | 0.100 | – |
| 50k | off track (262) | 0/10 | – | 0.065 | 265 |
| 100k | off track (449) | 0/10 | – | 0.072 | 248 |
| 150k | off track (478) | 0/10 | – | 0.072 | 242 |
| 200k | off track (478) | **2/10** | 115.5 s | 0.081 | 236 |

Training episodes ending at the time limit (cumulative): 7 by 50k, 55 by 200k. This is in line
with p1_TUM, whose first clean deterministic lap was 110.5 s at 150k. Healthy so far.

## Open points / caveats
- The keeper's stop rule (< 95 s) is below this car's QSS 102.5 s and near the OCP 94.6 s, so it
  may never trigger. The run then continues until stopped.
- Diagnostic thresholds in EvalConfig were tuned for the old car and tyre and are unchanged:
  `spin_slip` 0.12, `front_lock_kappa` -0.12, `car_half_width_m` 1.0 (sf70h_f1 is 1.035). They are
  logging only. Demo `brake_cap` stays 0.3 (4.5 kN·m at 15 kN·m), as in p1_TUM.
- The demo-dataset key hashes env.py, config.py and demo_driver.py. A fresh run of the old presets
  therefore recollects demos once (~1 min). The data is identical.
- `qss_profile.cache_key` does not hash `ocd_lap/cars.py` (known failing test). A later change to
  the f1 car values would not invalidate this QSS cache.
- Keeper log: harmless "evals.jsonl not found" tracebacks from before the first eval.
