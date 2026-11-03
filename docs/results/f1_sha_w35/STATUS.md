# f1_sha_w35: effect of the line (status 4 Nov 2026, 00:15)

Controlled experiment: run `f1_sha` (car sf70h_f1, setup p1_TUM_f1, time_pbrs from scratch, seed 0)
with ONLY the reference line changed: xi_ref = the converged (iterated-QP) minimum-curvature line,
width allowance 3.5 m (`docs/results/shanghai_line_setup/line_iqp_w3.5.csv`, ReferenceLine digest
`eaf1792a00be2da0`, CSV sha256 `db14a0ca…d180b3`, 5364.7 m, footprint fully inside, min clearance
+0.10 m). QSS p1_TUM_f1: **96.01 s on this line** vs 102.53 s on the safe line (Track.path).
f1_sha (safe line) was stopped at best 103.06 s.

Run `runs/mac/D2/f1_sha_w35/seed0`, preset `f1_sha_w35`, rlsetup commit `4533fe9` (branch sf70h-f1,
pushed, clean). ocd_first_lap = worktree `/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1`
(commit `ee86683`, clean) via PYTHONPATH.

## What changed (commit 4533fe9)
1. `EnvConfig.reference` accepts `'csv:<repo-relative path>'` -> `ReferenceLine.for_track` ->
   `from_csv` (path resolved against the repo root, not the cwd; a string, so it reaches the
   vector-env workers through the config dict). provenance.json gets a `reference` block (spec,
   CSV path + sha256, digest, length, QSS lap on the line, QSS lap on Track.path), only for csv
   lines, so 'path' / 'raceline' runs record exactly what they did before.
2. `DrivingEnv.drive_line` = xi_ref for every reference except 'raceline' (unchanged: Track.path).
   Start placement, settle, demonstrations and the evaluation start follow the new line.
3. Progress ds, the lap line and the QSS start-speed profile stay on Track.path (checked in
   env.py: `track.match` -> `_dist`; unchanged). `qss_profile.line_lap_time` computes the QSS lap
   on the line from the profile's g-g-v (96.0096 s, = lines.json 96.0096).
4. Preset `f1_sha_w35` = `f1_sha` + `env.reference='csv:docs/results/shanghai_line_setup/line_iqp_w3.5.csv'`
   (test: equal to f1_sha except the reference). Existing presets: same config, same drive line;
   the R1 golden-hash test passes. Note: the demo cache key hashes env.py / reference.py, so a fresh
   f1_sha run would recollect its demos (same data expected; not rerun).
5. `scripts/check_line_starts.py` (settle check along the line + demo dataset), `tests/test_line_reference.py`.

## Checks before launch
- Settle along the whole line: start every 100 m of Track.path at f = 0.3 / 0.5 / 0.7 ->
  **162/162 on track**, max |e_ref| at hand-over 0.60 m, max |track position| 0.77 (safe line,
  same check: 162/162, 0.80 m, 0.75). Evaluation start e_ref -0.003 m.
- Demo dataset (exact preset settings, cached and reused by the run, `demos_90ea46adaf5d`):
  142 episodes, **clean 0.894** (gate 0.8; f1_sha on the safe line: 0.921), causes off_track 15 /
  time_limit 127, mean speed fraction 0.634.
- Fast suite (PYTHONPATH = f1 worktree): 298 passed, 4 skipped, 2 failed = the 2 known failures
  (test_provenance cars.py not hashed; test_qss_profile p0 cache key). Before the change: 292
  passed, same 2 failures.

## Commands
```bash
caffeinate -dimsu nohup env PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 \
  platforms/mac/train.sh --preset f1_sha_w35 --seed 0 --run-name f1_sha_w35 > runs/mac/f1_sha_w35_stdout.log 2>&1 &
nohup runs/mac/keeper_f1.sh f1_sha_w35 > runs/mac/keeper_f1_sha_w35.log 2>&1 &
.venv/bin/tensorboard --logdir_spec "LINE_w35_new:runs/mac/D2/f1_sha_w35/seed0,LINE_safe_old:runs/mac/D2/f1_sha/seed0" \
  --port 6006 --host 127.0.0.1 --reload_interval 30
```
PIDs: training (python) 31649, caffeinate 31651; keeper 32057; TensorBoard 32063 (old one, 5099, stopped).
keeper_f1.sh exports the f1 PYTHONPATH, resumes `--preset f1_sha_w35 --run-name f1_sha_w35`
(up to 5 times) and stops at the first clean deterministic lap 2 < 95 s (threshold unchanged).
Stop by hand: `pkill -f "keeper_f1.sh f1_sha_w35"; pkill -INT -f "run-name f1_sha_w35"`.

## First evaluations (seed 0)
| step | det. | stoch. clean | stoch. mean lap [s] | alpha | Q_min / MC | env steps/s | training failures (cum.) |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 (BC) | off_track | 4/10 | 151.35 | 0.100 | 1.06 | - | - |
| 50,000 | off_track | 0/10 | - | 0.082 | 1.39 | 231 | off_track 97, time_limit 9, wrong_way 2, stall 3 |

f1_sha at the same points: 0/10 and 0/10 stochastic clean, Q_min/MC 1.08 / 1.89, a rollback at
40k (none here). BC gate: deterministic off_track, 0/3 stochastic (as f1_sha; gate not required).
Healthy: phase rl, running, 0 rollbacks, keeper mirroring evals into TensorBoard.
