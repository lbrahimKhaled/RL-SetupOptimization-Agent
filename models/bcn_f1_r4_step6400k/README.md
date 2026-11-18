# Barcelona F1 driver: `bcn_f1_r4` at 6.4M steps (picked 18 Nov 2026, Windows)

SAC driving agent (pure RL policy; pure pursuit only made the BC demos and the 2 s start hand-over) for
**sf70h_f1 on Catalunya**, setup **`bcn_f1_best`**, reference line **IQP w3.0**
(`docs/results/bcn_line_setup/line_iqp_w3.csv`), reward R4 from scratch + corner line penalty
(preset `bcn_f1_r4`, DECISIONS #28, notes `docs/results/bcn_line_setup/NOTES.md`).

## Why this checkpoint

Re-test on the Windows PC (`retest/`): each candidate drove the run's deterministic 2-lap evaluation and
30 stochastic laps (seeds 5000-5029, identical for all). Candidates = the 4 fastest checkpoints per run
whose own training evaluation had a clean deterministic lap and >= 8/10 clean stochastic laps.

| run (line) | step | det lap 2 | stochastic clean | stochastic mean |
|---|---|---|---|---|
| **bcn_f1_r4 (w3.0)** | **6.4M** | **85.33 s** | **29/30** | **87.28 s** |
| bcn_f1_r4_w4 (w4.0) | 3.2M | 85.87 s | 28/30 | 88.13 s |
| bcn_f1_r4 (w3.0) | 3.2M | 85.28 s | 27/30 | 87.87 s |
| bcn_f1_r4_w4 (w4.0) | 4.0M | 85.56 s | 25/30 | 87.71 s |
| bcn_f1_r4 (w3.0) | 7.0M | 85.15 s | 24/30 | 87.25 s |
| bcn_f1_r4_w4 (w4.0) | 3.6M | 86.46 s | 24/30 | 88.19 s |
| bcn_f1_r4_w4 (w4.0) | 1.1M | 87.18 s | 22/30 | 89.24 s |
| bcn_f1_r4 (w3.0) | 7.3M | off track | 20/30 | 87.18 s |

The one stochastic crash of 6.4M is at 3500 m (T10). QSS on this line and setup: 80.11 s (optimistic, ~3 s).

## Files

`seed0/` is a resumable seed folder: `checkpoints/last.zip` (= `step_6400000.zip`, sha256 3e8bea21...;
actor, critics, optimizers, log alpha), `config.json`, `provenance.json`, `status.json`, `evals.jsonl` /
`probes.jsonl` / `train_stats.jsonl` cut at 6.4M, `bonus_state.json`, `probe_starts.json`. **No replay
buffer** (1.3 GB, stays on the Windows PC).

## On the Mac

Continue OCD training (tested on Windows: resumes at 6,400,000 and trains):

    mkdir -p runs/mac/D2/bcn_f1_r4 && cp -R models/bcn_f1_r4_step6400k/seed0 runs/mac/D2/bcn_f1_r4/seed0
    PYTHONPATH=/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap-f1 platforms/mac/train.sh --resume \
        --preset bcn_f1_r4 --seed 0 --run-name bcn_f1_r4 --allow-code-change --allow-dirty

- `ocd_first_lap` must be on branch `sf70h-f1` (Windows used ee86683). `--allow-code-change` accepts the
  different code version (recorded in status.json).
- It starts with an **empty replay buffer**: no gradient step until 5,120 transitions, and the critic
  re-fits to fresh data, so expect a dip before it is back at this level.

Use the policy elsewhere (e.g. as the Assetto Corsa source actor):

    from rlsetup.driving.agent import StagedSAC
    model = StagedSAC.load('models/bcn_f1_r4_step6400k/seed0/checkpoints/last.zip', device='cpu')
    action, _ = model.predict(obs, deterministic=True)

- Observation: 159 floats from `DrivingEnv._observation` (`rlsetup/driving/env.py` docstring; scales in
  `config.json` env.obs): ego state, **e and heading error relative to the IQP w3.0 line**, track position,
  20 preview stations of that line and both track edges, last action, 3-decision history. A consumer must
  build it with the same line (`line_iqp_w3.csv`), not another reference such as `ocd_path`.
- Action: [steering, throttle, brake] in [-1, 1]; steering x 0.30 rad (sf70h_f1 lock); pedals with
  `pedal_deadzone` = true (pedal = clip((a + 0.5) / 1.5, 0, 1)); one action per 2 base ticks (25 Hz).
