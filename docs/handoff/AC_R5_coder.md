# AC fine-tune of bcn_f1_r5 (7M) with the R5 reward: implementation and launch files

20 Nov 2026. Nothing committed, nothing launched. The R4 learner (runs/ac/tcp-f1r4-001, port 2350) was not touched.

## Status
- Done: steps 1 to 6 (code, tests, R5 reference, v15 worker zip, Mac launcher, checks below).
- Not done:
  - The full repo test suite (`pytest -q` over all tests). It was started but had not finished when work
    stopped, so I have no result for it. Only the AC and R5 tests were run to completion (356 passed).
  - No commit.
  - No launch.
- Exact state of the changed files:
  - Modified, uncommitted: `rlsetup/assetto/driving.py`, `scripts/ac_ocd_reference.py`.
  - New, untracked: `tests/test_ac_r5.py`, `platforms/windows/ac/lenovo/START_LENOVO_F1R5.cmd`, this file.
  - New under runs/ (not shown by git status): the R5 reference, the v15 zip, START_MAC_F1R5.command.
  - `data/corner_table_276bbf9a604e.json` was already untracked before this work. The R5 reference
    generation needs it, so commit it with this stage.
  - `docs/research/full_throttle_sac.md` is untracked and was not written in this work.
- Next steps:
  1. `cd /Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent && PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 .venv/bin/python -m pytest -q`
     (full suite; compare against the known failures).
  2. Commit driving.py, ac_ocd_reference.py, test_ac_r5.py, START_LENOVO_F1R5.cmd and
     data/corner_table_276bbf9a604e.json. If the commit should match the zip, rebuild v15 after committing,
     so that BUILD.json git_commit is the new commit. The runtime sha256s stay the same.
  3. Launch as described under "How to launch".

## Files changed / added
- `rlsetup/assetto/driving.py`
  - `_profile_fields`: optional `profile.post_apex_w` (one value per row, in [0, 1], 0 on non-corner rows).
  - `Profile.post_apex_w(station)`: the row value, held over the row segment exactly like `corner()`, so w and
    the window flag always come from the same row. Rows are about 1.5 m apart on Barcelona and the blend is
    20 m, so one row step changes w by up to about 0.08. Raises if the reference has no field.
  - `RewardTerms`: new fields `apex_split, line_k_pre, line_free_pre_m, exit_k, exit_sigma` (off by default).
    Blended line penalty and the exit term, both once per 25 Hz decision at the decision end (like the R4 terms).
    `from_source` now fails on any source `env.reward` key AC does not know. Two groups of keys are accepted:
    keys that only matter in ref_tracking mode (`c, w, penalty_base, penalty_kappa, recovery_time_s`, which a
    time_pbrs reward never uses), and `apex_blend_m, split_units, split_at_m`. The second group is built into
    `post_apex_w`; without apex_split it must keep its neutral value (20, [], []). It also fails on
    `exit_k > 0` without apex_split, and on apex_split with `line_k = 0`.
  - `RewardTerms.identity()` replaces `asdict(terms)` in the chunk `source_terms`. Without apex_split the dict is
    exactly the old R4 one. With it, the dict also holds `apex_split='reference_post_apex_w'` (a string, because
    `learn.read_chunk` only takes numbers and strings) and the four R5 numbers.
  - `collect`: an apex_split source with a reference that has no `post_apex_w` raises.
- `scripts/ac_ocd_reference.py`
  - With `env.reward.apex_split`, adds `profile.post_apex_w` and `profile.post_apex` (blend, split map, rule).
    `post_apex_w` is `rlsetup.driving.env.post_apex_weight` at each row's s_path, using the env's split map
    (1-based units, checked to lie inside that unit's window), and 0 outside the windows.
  - QSS profile: the R5 run recomputed its QSS profile in memory. It ran on the Mac with ocd_first_lap-f1, and
    the only data/ file has the Windows R4 key c6cc22. So no data/ file carries the R5 key 6e00b6, and the old
    script stopped there. It now recomputes the profile with `qss_profile.compute`, as `load_or_build` does on a
    key mismatch, but only if the current cache key equals the provenance key. It also checks the key of the
    result. The source block then has `qss_file: null` and `qss_recomputed_key_sha256`. Nothing is written to data/.
  - R4 output is byte-identical: regenerating the R4 file gives sha256 85e6fa8a..., the same as the file in v14.
- `tests/test_ac_r5.py` (new, 56 tests).
- `platforms/windows/ac/lenovo/START_LENOVO_F1R5.cmd` (new, CRLF line endings).
- `runs/ac/ocd_line_reference_bcn_f1_r5_7m.json`, `runs/ac/lenovo-tcp-f1r5-v15.zip`,
  `runs/ac/tcp-launch/START_MAC_F1R5.command` (chmod +x).

## Tests (PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1)
`.venv/bin/python -m pytest -q tests/test_ac_*.py tests/test_r5.py` -> `356 passed, 2 warnings in 15.56s`
(the warnings were there before: a torch scalar warning and the Shanghai QSS cache key).
- R4 bit-identity: the v2 harness drive (R4 source, 172 decisions) gives the same reward bytes and the same
  `source_terms` before and after the change.
- The R5 terms equal `DrivingEnv._line_penalty` / `_exit_term` at w = 0, 0.5, 0.75 and 1, for 4 |e| values,
  ds > 0 / 0 / < 0 and rho from 0.6 to 1.2. A full AC drive with R5 terms matches the spec formulas written
  out in the test (rtol 1e-5), and `read_chunk` accepts the resulting chunk.
- Reference: w = 0 at 875 and 920 m, 0.5 at 930 m, 1 at 940 and 980 m in U1, on the corner table and in the
  generated file (the row nearest 930 m is at 930.78 m, w = 0.539). A missing field raises, an out-of-range
  field raises, and an unknown reward key raises.

## R5 reference vs R4 reference
`runs/ac/ocd_line_reference_bcn_f1_r5_7m.json` (sha256 0633b358...). Same AC track file
(runs/ac/ac-track.json, sha256 2a692f4c..., as the R4 file's `ac_track_file_sha256`), same arguments.
- Identical: `rows` (2988), `alignment`, `diagnostics`, `settle`, `profile.s_path_m`, `profile.corner`.
- Not identical: `v_qss_mps` (max |d| 0.67 m/s at s = 1129 m, mean 0.013 m/s) and `v_start_mps` (max 0.19 m/s).
  The reason is that R5 uses its own corner table, `corner_table_276bbf9a604e.json` (untracked in data/). The
  R4 table 6313f2151e24 was built on the Windows PC, and the two tables were built from different Track.path
  hashes and OCD builds. U2's window starts 1 m earlier (1130.9 vs 1131.9 m). Apexes and the other windows
  agree to 1e-11 m. No row's corner flag changes.
- Source binding: run, config sha256 (39ea7d22..., equal to the actor bundle's), corner table, and QSS
  (recomputed, see above). New fields: `post_apex_w`, `post_apex`.
- `post_apex_w`: 773 rows with w > 0, 673 with w = 1, w = 0 on every non-corner row.

## Worker package v15
`runs/ac/lenovo-tcp-f1r5-v15.zip` (sha256 73edaa83...) uses the v14 layout under `ac-lenovo-tcp/`:
- BUILD.json (schema v1, version 15, git_commit dd122ee, runtime sha256s, reference sha256);
- the 10 runtime files (no __pycache__, no learn.py);
- tcp-session.key, copied from runs/ac;
- the R5 reference, START_LENOVO_F1R5.cmd and CALIBRATE_F1R4.cmd (unchanged).

The BUILD change line says that driving.py has uncommitted changes on top of dd122ee.

Isolation check: the package was extracted to an empty folder and run with `python -I`. The runtime hashes
match BUILD.json. The reference was loaded on the real AC track data, Profile and RewardTerms were built from
the R5 bundle config, and the binding and lock checks passed. After all of that, none of `rlsetup.driving*`,
`rlsetup.assetto.learn`, `ocd_lap`, `ocd_native`, `stable_baselines3` or `gymnasium` had been imported.

## Mac server check (no port bound)
A `LearnerServer` with exactly the START_MAC_F1R5 arguments was built in a scratch work dir. It wrote
server.json (critic_init_sha256 d4f993e2...) and unpacked the bundle. `build_learner` then loaded the critics
and alpha from `models/bcn_f1_r5_step7000k/seed0/checkpoints/last.zip`: the checksum matches the bundle
manifest, the actor tensors are equal, and log_alpha = -2.764. A `hello` returned cycle 0. During this check
the R4 learner was still the only process listening on port 2350. runs/ac/tcp-f1r5-001 was not created.

## How to launch
1. Mac: stop the R4 learner first (both learners use port 2350). Then run
   `runs/ac/tcp-launch/START_MAC_F1R5.command`. It creates the new work dir runs/ac/tcp-f1r5-001.
2. Lenovo: unzip lenovo-tcp-f1r5-v15.zip.
   - If you unzip into a new location, copy `steering-f1r4.json` from the old ac-lenovo-tcp folder: same
     bcn_f1_best setup. Run CALIBRATE_F1R4.cmd only if that file is missing.
   - Unzipping over the old folder keeps steering-f1r4.json and .venv.
   - Optional: copy `tcp-worker-f1r4-001\heights.npz` into a new `tcp-worker-f1r5-001\` folder first. The
     reference rows are identical, so the height map stays valid, and teleport starts work from the first drive.
   - Run START_LENOVO_F1R5.cmd. AC must be on track with bcn_f1_best and the CSP rl_teleport app installed,
     as for F1R4.

## Points to confirm
- START_LENOVO_F1R5.cmd is based on the current `platforms/windows/ac/lenovo/START_LENOVO_F1R4.cmd`, which has
  `--teleport-dir` (AC_TELEPORT_DIR). It is not based on the older copy inside v14, which has no teleport. I
  chose the newer one because the running R4 run uses teleport starts (progress.jsonl shows teleport retry and
  random starts), and the v15 runtime has the teleport code. To drop teleport, delete the `--teleport-dir`
  argument.
- The exit term and the w blend are applied once per 25 Hz decision at the decision end, while OCD applies
  them per 50 Hz tick (the same approximation as the R4 terms).
- In `env.corner_bonus`, the R5 entry/exit closeness bonuses are 0, so there is nothing to port. The AC path
  never implemented R4's bonus_in/bonus_out = 10, so the R4 AC run already lacked them. This was already the
  case and is unchanged here.
