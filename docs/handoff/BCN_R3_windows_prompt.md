# Prompt for the Windows machine: Barcelona with the R3 corner bonus (8 Nov 2026)

Paste everything below the line into the Windows session (repo RL-SetupOptimization-Agent).

---

We are starting a Barcelona (Catalunya) run on this Windows machine with the F1 car `sf70h_f1` (NOT the
AV-21-based `sf70h_tum`) that does what the Mac is doing on Shanghai right now (run `f1_r3`). Work through the steps in order, with the usual team pipeline
(rl-architect approves → rl-coder implements → qa-reviewer + sim-runner verify → you report to me → commit).
Save every subagent report to a file under docs/ and commit it. Commits are brief, by me, with no trailers.
Before anything that changes the plan below, or any OPEN item, ask me.

**0. Get the Mac's work.** `git fetch origin`, then `git checkout sf70h-f1` and `git pull` (head at least
`dc36e79` "Handoff: f1_r3 running"). Read first:
- `docs/DECISIONS.md` #21–25;
- `docs/specs/R2_review.md` (with Amendments 1–2), `docs/specs/R3_bigger_bonus.md`, `docs/specs/R3_review.md`;
- `docs/handoff/R3_coder.md` and the 8 Nov sections of `docs/handoff/HANDOFF.md`;
- `docs/results/shanghai_line_setup/NOTES.md` (how the Shanghai line and setup were made) and
  `docs/results/bcn/SETUP_SELECTION.md`.

The F1 car `sf70h_f1` needs the simulator branch `sf70h-f1` of ocd_first_lap (commit `ee86683`, pushed).
- Check whether this machine already has a built worktree of that branch.
- If not, add one next to the existing checkout and build `ocd_native` with the MinGW recipe in
  `platforms/windows/README.md`. The Windows build fixes live on `sf70h-tum-car` (3945634). If the f1 branch
  doesn't build, tell me before changing anything in ocd_first_lap.
- Point `OCD_FIRST_LAP` at that worktree for every F1-car command.
- Run `python -m pytest -q -m "not slow"` and compare against the known Windows failures
  (`docs/handoff/W1_windows_port.md`) plus the 2 known ones (test_provenance cars.py, test_qss_profile 'provenance').

**1. Inventory the existing Barcelona runs on this machine** (`runs\windows\D2\*`, `runs\D2\*`). For each one:
- run name, preset, car, setup, reference line, env steps, state, best clean deterministic lap 2;
- whether `checkpoints\last.zip` and `replay_buffer.pkl` exist.

Report this to me before step 3.

**2. Barcelona prep, as was done for Shanghai** (my decision):
- **(a) Setup sweep:** a QSS setup sweep for `sf70h_f1` on Catalunya, the same way as `sha_f1_best`
  (`scripts/sha_line_setup.py`, NOTES.md). The result is a new named setup (proposed name `bcn_f1_best`).
- **(b) Line:** the iterated-QP minimum-curvature line with a 3.5 m width allowance on Catalunya, as Shanghai's
  `line_iqp_w3.5.csv`. Generalise the script to take the track. Results go in `docs/results/bcn_line_setup/`.
- Report the QSS lap on the safe line and on the new line, and the corner table that `rlsetup/driving/corners.py`
  detects on it. Its detection parameters (prominence 5 m/s, merge 150 m, exit = apex + 150 m or 30 m before the
  next braking) were tuned on Shanghai, so the architect must check that the table makes sense on Catalunya.

**3. Barcelona run on the F1 car only, with the bonus on from the start** (my decision: car `sf70h_f1`, the
non-AV-21 car with the PL14 tyre and literature parameters; **never `sf70h_tum`**; no plain phase first).
The source run depends on what step 1 finds:
- **A Barcelona run on `sf70h_f1` exists:** continue it as `<source>_r3`.
  - Warm start from its **`last.zip` + replay buffer**, not `best.zip`. On Shanghai a best.zip critic didn't
    match the end-of-run buffer and caused new crashes.
  - Use the new setup (2a) and line (2b). The architect confirms the warm start across the setup/line change:
    the critic must re-fit, and the buffer is relabelled to the new reward.
- **No `sf70h_f1` Barcelona run exists** (the AV-21 / sf70h_tum runs don't count, leave them alone): start
  `sf70h_f1` **from scratch** on Catalunya as run `bcn_f1_r3`.
  - Use the `f1_sha` recipe (demos + BC init, time_pbrs, 15 kN m, traction-control demo settings of
    sf70h_f1) with setup `bcn_f1_best` and the 2b line.
  - Turn the R3 aids on from step 0.
  - Focused starts begin uniform and adapt from training crashes, since there is no crash history to seed from.

The R3 aids, as in f1_r3:
- clean-corner exit bonus B = 30 per corner, floor 0.85, capped at v_qss on the reference line, paid only if
  the car stays on track through the corner;
- **kept on (w = 1), never fades**;
- focused starts 25 %;
- low-noise rollouts.

For a warm start, use the f1_r3 warm procedure: offline critic 50k grad steps with the pre-exit probe, online
critic 20k, then unfreeze, actor lr 1e-4, guard 0.65, lifetime step offset chained from the source.

**Windows-specific items for the architect:**
- n_envs 16 (`train.overrides`): `low_noise_envs` should keep about a quarter of the envs, e.g. 4 of 16, not
  only env 3.
- Eval, guard and checkpoint intervals are in env steps and stay as on the Mac.

**4. Launch and watch.**
- Launch detached with `platforms\windows\start_detached.ps1` (`nohup &` dies on this machine).
- Write and commit a Windows keeper (`platforms\windows\keeper.ps1`), a port of the Mac's keeper logic:
  - mirror `scripts/evals_to_tb.py` every 3 min;
  - stop at the first clean deterministic lap 2 ≤ the limit (`scripts/keeper_check.py reached`);
  - resume from last.zip if training dies (at most 5 times);
  - never resume a run that ended itself.
- **Stop limit for Barcelona:** propose the same ratio as Shanghai, 95 s / 93.83 s QSS on the line = 1.0125 ×
  the QSS lap on the Barcelona line. I decide the number.
- To stop by hand, stop the training only right after a checkpoint and buffer save. On the Mac a backgrounded
  trainer ignored SIGINT.
- TensorBoard: the new run's `tb_evals_lifetime` next to its source run.
- Report to me after the step-0, 20k and unfreeze evaluations:
  - critic_check and the pre-exit Q;
  - deterministic and stochastic clean rate;
  - corner ρ.

Then report every ~1M steps: ρ per corner, crashes per corner, best and median clean lap 2.

Don't push run folders (`runs/` is gitignored); push code and docs on `sf70h-f1`.
