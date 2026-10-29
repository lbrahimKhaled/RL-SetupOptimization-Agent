# Training stopped by explicit user request

Stop record created: 2026-10-29T09:50:36.422316+00:00. Both restart keepers were stopped first. Learners received exact-PID SIGINT, then SIGTERM after cleanup did not finish promptly. No process-group signal was used. No learner, captured worker, or restart keeper remains.

## Local resume state

Keep the entire original `runs/mac/D2/p1_TUM/seed0` and `runs/mac/D2/p0_TUMref/seed0` trees. Checkpoints and replay buffers remain in those ignored local run directories; they were not copied into this commit-able record. The two replay buffers occupy about 2.6 GB together. This record alone cannot resume training on another machine. All original logs and periodic checkpoints remain intact.

- `p1_TUM`: resume from **10,500,000** model steps; replay saved at **10,500,000**. Best clean deterministic lap 2 **101.269925406 s**, checkpoint step **9,600,000**. Latest logged step **10,516,952**; at least **16,952** logged later steps are outside the saved model. Additional unlogged tail is unknown.
- `p0_TUMref`: resume from **4,950,000** model steps; replay saved at **4,950,000**. Best clean deterministic lap 2 **103.368638555 s**, checkpoint step **4,900,000**. Latest logged step **4,973,784**; at least **23,784** logged later steps are outside the saved model. Additional unlogged tail is unknown.

Both preserved `status.json` files actually still say `running`; that historical field is stale after the OS-level stop. Exact learner, worker and keeper process checks found none remaining. The observed stdout tails end in ordinary training metrics, with no final-save confirmation or KeyboardInterrupt traceback. SIGINT did not finish shutdown within the short window, so exact-PID SIGTERM completed the intentional stop. No original status or log file was edited. SIGINT does not save final in-memory progress. Resume loads the saved model/optimizers/replay and starts new environment episodes; it does not restore an in-flight physical lap. The replay-save callback can omit the current four-environment vector transition at its save boundary, as documented in the training code.

## Resume later

From the repository root, run only when the user decides to resume:

```sh
platforms/mac/train.sh --resume --preset p1_TUM --seed 0 --run-name p1_TUM --allow-code-change
platforms/mac/train.sh --resume --preset p0_TUMref --seed 0 --run-name p0_TUMref --allow-code-change
```

The launcher retains each saved device/configuration (p1 CPU, p0 MPS). Do not add device overrides on resume. `--allow-code-change` is included because a later commit changes the code identity; resume records that mismatch. Commit intended tracked changes before resuming a strict-provenance run. The restart keepers remain off; the commands above do not launch them.

`stop_resume_manifest.json` records exact launch commands, resume commands, code identities, SHA-256 hashes of every checkpoint and both complete replay files, matched-step checks and original paths. Small config/provenance/status/evaluation/progress copies and runtime launch scripts are stored alongside it. No ZIP, replay buffer, or model weights should be force-added to Git.
