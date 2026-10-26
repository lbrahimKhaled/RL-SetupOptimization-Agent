# D2 QA review — qa-reviewer (26 Oct 2026)
Verdict: **no BLOCKERs; 2 MAJOR, 7 MINOR.** Reviewed ef2c736..fa4eb97. Tests: 118 passed (152.6 s, contended). Throwaway checks in /tmp/qa_d2 (temporary).

## Verified at runtime
1. StagedSAC freeze vs SB3 2.9.0: 50 frozen train() steps → actor + log_ent_coef bit-identical, log_ent_coef.grad None; actor/α Adam state empty → fresh at unfreeze; ent_coef_optimizer restored if train() raises; actor_lr override works; save/load keeps optimizer states bound; continued training original vs loaded bit-identical; saved-while-frozen stays frozen.
2. Prefill vs NStepReplayBuffer (buffers.py:886–950): column tails done=1 + timeout=1; never chains into online rows; terminated/truncated preserved.
3. Determinism: two same-seed CLI mini pipelines bit-identical (BC sha, jsonl minus wall-clock, last.zip policy).
4. Conventions: lap indexing matches env.py:318–365; edge-margin formula correct (track_position −1 right / +1 left); front-lock sign correct (kappa negative on lock).
5. BC σ [0.052, 0.287, 0.413] not a bug (pre-tanh σ = action noise/(1 − a²); demo pedals far from 0).
6. BC loss and output-bias init match spec ([0, −0.2822, −1.0986], σ ≈ 0.37).
7. CollapseGuard snapshot covers actor, critic, target, all 3 optimizers, log_ent_coef; in-place restore; snapshot before any actor update.

## MAJOR
- **M1 (spec-level → architect):** Q-vs-MC diagnostic biased by SB3 n-step target (entropy enters only every n-th decision). evaluate.py:198–208, used evaluate.py:328, train.py:188. Offset = α·E[logπ]·(γ/(1−γ) − γ³/(1−γ³)) ≈ α·logπ·83 → +24.9 at α = 0.1, logπ = 3 (true ~57) → reads as ~44 % "overestimation" and drifts with α. Fix: n-step-consistent MC (entropy at j ∈ {n, 2n, …}) or log both; test vs brute-force fixed point. Also: n-step SAC in SB3 applies only ~1/3 of the critic-side entropy bonus.
- **M2:** resume neither records nor checks code provenance (train.py:329–383; dirty check only on fresh start, train.py:213). Fix: on resume compute git_state + ocd_lap_code_version, store in resume event, refuse on mismatch vs provenance.json or dirty full preset unless --allow-code-change / --allow-dirty (recorded).

## MINOR
- m1: resume chains n-step targets across the resume cut (train.py:372): 69/12,800 samples crossed episodes. Fix: after load_replay_buffer set rb.dones/timeouts[(pos−1) % size] = 1.
- m2: d2_report crashes on a seed without last.zip (d2_report.py:56). Fix: mark FAIL with status reason.
- m3: crash right after a rollback desyncs rollback state (callbacks.py:145–149). Fix: atomic last.zip on rollback or keep count on model.
- m4: hard-kill resume can pair newer last.zip (every eval) with older buffer (every 100k). Fix: save buffer at evals too, or warn when replay_buffer_step < num_timesteps − n_envs.
- m5: lap-2 metrics tagged at decision granularity (evaluate.py:134), < 40 ms shift. Document.
- m6: TrainStatsCallback thresholds hardcoded (callbacks.py:158) → EvalConfig.
- m7: run_d2_full.sh skips seeds done by wall_clock_limit → "+50 % resume" must be manual.

## Mutation testing (46 mutants): 25 killed, 5 equivalent, surviving = test gaps to add
A3 (agent.py:50 freeze condition swap); B1, B3 (bc.py:85, 83 best-val/early-stop); B2 (bc.py:22 actions at ±1 finite loss); E1, E2, E7, E8 (evaluate.py:328, 336 Q/obs pairing, failure tail, entropy, α source); E4 (spin upshift across lap boundary); C1 (guard at exactly 0.5 × baseline passes); T3 (train.py:391–394 resume during frozen warm-up stays frozen); R1, R2, R3, R5 (d2_report.py:112–114, 152 boundary values); V1 (track width at start); plus resume-chain test (m1) and gate-failed-seed report test (m2).
