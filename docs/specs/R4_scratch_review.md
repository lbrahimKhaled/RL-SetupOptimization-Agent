# Review of preset f1_r4_scratch (rl-architect, read-only, 9 Nov 2026)

**Verdict: CHANGES-REQUIRED.** Reward wiring, demo labelling, n-step accounting and terminal convention are sound. Three things must be settled before launch; none reopens DECISIONS #27 or the reward design.

| Layer | Verdict |
|---|---|
| Preset vs control config | APPROVE |
| Env / reward path (env.py, speed_profile.py) | APPROVE |
| Demo labelling and prefill (demos.py) | APPROVE-WITH-NOTES |
| n-step accounting (buffers.py), w(0) fix (train.py) | APPROVE |
| Guard and actor learning rate (comparability) | CHANGES-REQUIRED (B1) |
| Launch and keeper procedure | CHANGES-REQUIRED (B2, B3) |

## Blockers
**B1. The actor learning rate depends on whether the guard fires.** `CollapseGuard` sets `model.actor_lr = rollback_actor_lr` only on a rollback (`callbacks.py:200`); in the BC path nothing else sets it (`warm.actor_lr` is applied only in `warm_start`, `agent.py:275`). The control rolled back at 30k (400.8 m vs limit 450.4 m), and all 247 evals from 50k record actor_lr 1e-4, so it trained at 1e-4 from 30k to 12.35M. If no probe fails in the 200k guard window, f1_r4_scratch keeps 3e-4 for the whole run; a later failure switches at a different step. Options:
- A (recommended): make the BC path honour `warm.actor_lr` and set it to 1e-4 in this preset, so the actor uses 1e-4 from the unfreeze. Matches the control on 99.8 % of steps; thresholds and rollback logic unchanged.
- B: no code; at 220k, if `rollbacks` is 0, stop and relaunch with A.

**B2. With the keeper running, `collapsed` does not end the run.** `_learn_and_finish` saves last.zip with the collapsed weights, `keeper_check.py stopped` only recognises `report_complete` or a trigger, and `resume_pipeline` does not refuse `collapsed`. The keeper would resume up to 5 times from collapsed weights. Fix: start the keeper only after step 220k, or make `stopped` true for state `collapsed`.

**B3. Launch command omits the step cap.** Must include `--set total_timesteps=12350000` on the first launch (resume reads config.json).

## Answers
**Comparison validity:** valid apart from B1. Preset + Mac overrides vs control config: shared keys differ only in `preset` and `runs_root`; added keys are the reward items plus inert ones. Code since 78d4cb6: env.py draws no extra RNG with focus off, reward adds only shaping_v; `_sample_action` unchanged without low-noise envs; evaluate.py logging only; `CollapseGuard` no removed lines. Control's resume at 50k moved 78d4cb6 → 4cbea90 with an empty rlsetup diff. Simulator ee86683, clean. Corner table lap_time_s 93.825.

**Demo labelling:** 71 nonzero rows of 200,575 is correct labelling (g = 0 at or below 0.7 v_qss; control dataset mean speed fraction 0.6265). BC never uses rewards. Where both row ends have g = 0, shaping is exactly 0. Offline critic consistent on demo support (smoke Q/MC 1.005). Above rho 0.7 it may overestimate faster states by up to about k_v g (at most 20); bounded, smooth, transient. The 20k frozen online warm-up stays at BC speeds, so it does not correct this before the unfreeze; check via q_vs_mc.

**Guard:** control hit it once (30k), no collapse; the other 18 probes passed, lowest 494.4 m at 40k. Not materially riskier on this (weak) evidence: smoke/control 463/494 (40k), 862/662 (50k), 783/868 (60k). Keep threshold 0.5, baseline, interval, window and max_rollbacks 1 as in the control. Pre-register: a collapse is not a result (no lap data before 220k); relaunch once with `guard.max_rollbacks=2` (fair: the control had no second failed probe, so its trajectory is identical under that setting). Depends on B2.

**Nothing else invalidates the run:** Phi_e and Phi_v are 0 after off_track, nonfinite and wrong_way (`env.py:596-608`), truncation bootstraps; take-backs sit in the failing row and the n-step mask includes it; `_get_samples` adds w b per 1-step reward before discounting; w(0) set after prefill and before `_offline_warmup`.

## Notes (not blockers)
- Metric noise: the control has only 1–6 clean deterministic laps per 1M bin, medians 101.87–102.64 s after 5M. Report n per bin and use `sto_clean_rate` for clean rate.
- Compare `q_vs_mc` and probe progress with the control at 50k–250k (handover overestimate).
- Watch `lap2_r4` full-throttle share per straight and entry rho (new run only).
- Strict provenance applies to resumes: the tree must stay clean for the whole run (~14 h at the control's rate) or a keeper resume is refused with exit 4. Untracked files did not count as dirty for the control.
- After a completed run the keeper makes ~5 "nothing to do" resumes, then gives up.
- Compare the run's `ocd_lap_code_version` to 0efddcc91634 at launch.

## Verification status
Read or run: diffs of env.py, agent.py, evaluate.py, vec.py since 78d4cb6, removed lines of callbacks.py and train.py, the uncommitted diff, control status/probes/evals/checkpoint metadata, a config comparison. Not verified: test suite and smoke (coder's report only), demo obs/actions equal to the control's, learning behaviour, resume of a BC + R4 run.
