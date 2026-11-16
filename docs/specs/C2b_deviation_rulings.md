# C2b deviation rulings (rl-architect, 16 Nov 2026)

Scope: coder items 1–32 in `docs/handoff/C2b_coder_report.md`, checked against `docs/specs/C2b_addendum.md` (B1–B23), spec §12 (incl. 12.21), `rlsetup/critic/{train,meta_prior,metrics,evaluate}.py` and `scripts/critic_c2.py`. I reviewed design intent only; QA is checking correctness. Evidence tags: [C] read in the code, [F] formulation or decision, [J] my judgement. No literature is relied on.

**Summary:** 30 of 32 items are accepted, some with notes. One item needs a change (30), and item 29 is settled by amending B23. One more change, not on the coder's list (the R5b comparator), must go in before R2. After both, R2–R9 start in a **fresh run directory**, not `20261116_075115Z`.

## Per-item rulings

| # | Item | Ruling | Reason |
|---|---|---|---|
| 1 | B1 pass 2 on the rows pass 1 kept | **ACCEPT** | The B1 text ("drop again") means trimming in sequence, and a row dropped in pass 1 stays dropped. My script that re-tested all rows was the deviation. B1 counts amended to: rows A 20,190; rows S 1,633; dropped 71 low + 1 by the hard cut, 444 high. Exact rows (586/579) and Barcelona rows (977/24) are unchanged. Keep `label_retest_all=False`. |
| 2 | Order: box → hard cut → label filter | **ACCEPT** | The label median has to be taken over the rows that are actually used, and E1 measured its counts on rows that pass the box. Putting the hard cut before the median filter keeps implausible laps out of m and d. |
| 3 | Closed-form nuisance refit after every fit (critic frozen) | **ACCEPT** | [C] After the refit, every model is scored the same way on the same training rows: nuisance-only, ridge, critic 3, c3E/c2preE and critics 1/2 all get the penalised least-squares optimum of the B7 terms given their own setup part. The R² gain then measures what the setup part adds, and the measured −0.066 → +0.005 (ridge +0.004) shows the joint nuisance was a measurement artefact. q, the wd/c/L2/level selection (all on pre-refit inner-validation MSE) and the pair metrics are unaffected. The report must say: "R² uses the closed-form nuisance refitted on all training rows; it never enters q". |
| 4 | Exact sparse direct ridge solve | **ACCEPT** | It is the same estimator solved exactly, which is better than backfitting that does not converge. |
| 5 | Patience 30 (B8); smoke settings | **ACCEPT** | B8 overrides the C1 amendment. |
| 6 | wd and L2 grids selected per fit on that fit's inner validation | **ACCEPT** | This is what B8/B9 mean, and it is honest nested selection. |
| 7 | CV metrics use the twin mean; q_min only in the bundle | **ACCEPT, with a note** | The twin mean is the lower-variance estimate of the critic's ranking quality. The twin-0 block is already reported (`critic3_twin0_*`). Treat "bars are judged on the twin mean" as pre-registered. Optional and report-only: q_min pair accuracy, computed afterwards from the saved per-fit npz files. |
| 8 | Pairs scored on the setup part (J + Barcelona head, including λΠ); Spearman on the full prediction | **ACCEPT** | The setup part is what TD3 ranks [F]. [C] Ridge pairs are scored on the same basis (its p, p×z, pd terms) and on identical pairs (`Evaluator.pairs`). Date and team effects inside a pair are label noise for this metric, not a bias, unless setups drift with date. That risk is watched with the same-month pair accuracy, computed afterwards in the report. The report must label the Spearman as "full prediction (lap-time predictor), not setup ranking". |
| 9 | Barcelona head optimiser uses the B8 settings | **ACCEPT** | B11 left the optimiser open, and B8 is the natural default. |
| 10 | R6: real fit's L2 and its twin-0 bootstrap; train and test shuffled separately; m(s̃) gets the real s̃ | **ACCEPT** | The shuffled model keeps the same capacity and only loses the information, which is what a permutation test needs. Separate shuffles stop style values leaking between train and test. |
| 11 | R2 uses one non-bootstrap critic 3 per fold | **ACCEPT** | This is a cleaner comparison of the two grouping schemes. |
| 12 | Strict and unguarded variants use the twin-0 configuration | **ACCEPT** | The comparison is paired, so only the variant differs. |
| 13 | Leave-Barcelona-out and Barcelona-only critics | **ACCEPT, with a note** | [C] The code trains leave-out on **all** non-Barcelona rows (2 fits, matching the run table), not "of the fold" as the report says. That is fine because no Barcelona row is ever in training. The report must say leave-out has about 25 % more off-track data than a pooled fold fit, so pooled − leave-out is conservative. Correct the wording in the coder report. |
| 14 | R9 style reference read with `git show` at the pinned commit | **ACCEPT** | This gives reproducible provenance. |
| 15 | New `TrackStyleTransform` class | **ACCEPT** | The C2a tests stay valid. |
| 16 | MDE = 2.8 × SE | **ACCEPT** | 80 % power, two-sided 5 %. |
| 17 | `person` stored as integers; folds.json stores positions only | **ACCEPT** | Meets B6 privacy. |
| 18 | Behaviour on tiny fixture data | **ACCEPT** | Only test fixtures can reach it. |
| 19 | Pooled fast set = median of the union of the game's per-(game, track) fast sets; unseen track → Π = 0 | **ACCEPT** | This is the literal B21.1 text. Knobs where the meta matters (camber, toe) are global anyway (E21.2). |
| 20 | Weights differ from E21.10 because the coder pools small groups where my script skipped them | **ACCEPT: the coder is right** | B21.2 step 2 computes m on each half "by B21.1", so the fallback applies inside the halves too. Halves have about half the rows, which is why groups fall below 20 there. My note that "the smallest fast set is 28" was measured on all rows, not on fold halves; that was my error. The E21.10 numbers are superseded by the coder's (toe F 0.11, LSD coast 0.05). Pooling is also better than skipping: pairs on small tracks still count, and κ = 300 dominates. The pooled fallbacks for F1 25 Madrid and F1 26 Imola are expected and flagged. |
| 21 | Adoption item 3 (Barcelona) judged on F_cell; F_person reported | **ACCEPT** | This matches E4/R4, where the Barcelona protocol uses cell folds. With a Barcelona MDE of about 0.13, a 0.01 rule on both fold sets would only add noise. |
| 22 | B21.5 trigger: a miss on F_cell or F_person (plus strict on F_cell r0) | **ACCEPT** | R11 does not name a fold set. Reading it as "either" is the conservative choice for "fails to grade", and F_person is the OCD/AC situation. A false trigger only costs R3c, and delivery is still gated by non-inferiority. Note: if a fold set's modal c is 0, its F_person non-inferiority Δ is exactly 0 by construction. Report that case as "no prior on this fold set", not as a pass. |
| 23 | Published sensitivity at c = 0.03 when the modal c is 0 | **ACCEPT, with a report note** | A run at c = 0 would be identical to plain and therefore uninformative. But `delta_pair_acc_A_vs_in_fold` compares published at c* against the in-fold **selected mixture**, which mixes the leak with λ. The report must quote published − in-fold at the **same c** (c3pub vs c3meta_c{c*}; both fits exist in R3b). Compute it afterwards from the npz files; no change to the run code. |
| 24 | R3c with modal c = 0 reuses R3 | **ACCEPT** | B21 at c = 0 is critic 3 bit for bit. |
| 25 | R9 final c = modal over all R3b folds (tie → smaller) | **ACCEPT** | R3c evaluated the modal c, so R9 delivers the configuration that was evaluated. Also report R3c's Δ table on adoption; the adoption decision itself stays with R3b, as written. |
| 26 | S22a/S22b compared against plain R3 twin 0 | **ACCEPT** | The session questions are separate from Π; comparing against plain isolates them. |
| 27 | R8 stays on plain critic 3 | **ACCEPT** | R8 asks whether provisional z distorts the setup effects, and Π does not depend on z. |
| 28 | R6 ablation only on the rung-0 (B9) critic 1 | **ACCEPT** | B23 rung 3 fixes the statement for rungs 1–2 ("effect not detectable…"). A style-helps claim from a rung that exists because style did damage would not be coherent. Rows S get their own R6 through `--rows S`. Label R6 as "B9 critic 1" in the report. |
| 29 | Critic 2 rung 2 has code but no scheduled fits | **ACCEPT: do not schedule; B23 amended** | Critic 2 is not Q, and B23 already names its substitute (screening uses critic 1). Adding fits now means new code and QA before the run, and any later addition would invalidate the cache (see "Must change", point 3). Amendment: in C2b, critic 2 gets rungs 0–1 only. Its rung 2 is deferred to the OCD/AC stage. If it fails rungs 0–1, it is reported as a negative result and screening uses critic 1. Budget: 0. |
| 30 | B23 level selected per fit; bars tested on the resulting mixture; R9 picks the level by the twins' mean inner-validation MSE | **CHANGE** | Selecting per fit and testing the mixture is correct: that is nested-CV evaluation of the procedure R9 runs. Choosing one level per fold set would need outer information or a vote. **But the candidate set must be {L1, L2, L3}, not {B9, L1, L2, L3}.** Rung 2 is only reached after B9 failed the outer bars. With B9 in the set, R9 can deliver the configuration that just failed (it picks B9 whenever B9's inner-validation MSE is lowest), which breaks §12.21 ("make critic 1 safe"). The fix is in "Must change", point 2. This overrides the "{B9 setting, L1, L2, L3}" wording in B23. |
| 31 | Rung 1 only when the B16 trigger fires | **ACCEPT** | B23 says "same trigger". Rung 0 can fail its CI bar without the point-estimate trigger firing; it then goes straight to rung 2, as written. |
| 32 | Output keys avoid personal names | **ACCEPT** | |

## Must change before R2

1. **R5b comparator must use the same m(s̃) penalty as the level (B23 bars).** [C] `run_R5b` judges ΔR²(c1 − c3) against R5's `c3E`, which is the exact-row nuisance at α_style = 10. The rung-2 levels refit m(s̃) at α = 100, 300 and 1000 (`finetune(level=…)` → `refit_nuisance_ft` with the level's cfg).
   - m(s̃) is training-only and never in q [F]. E3 showed the style main effect hurts unseen drivers (−0.071 at α = 10).
   - So rung 2 can pass the bar by shrinking a nuisance term TD3 never sees, while the style path in the heads still does damage.
   - **Fix:** in R5b, add a comparator per level, `c3E{L}` = `finetune(d, etr, f3, 'c3_exact', cfg with nuisance.alpha_style = level α)`, per fold, repeat and twin. These are closed-form fits, about 0.02 s each. Judge each level, and each fit of the "selected" mixture, against its matched comparator. Keep the comparison against the α = 10 comparator in `metrics_R5b` as report-only.
2. **Rung-2 selection set = ('L1', 'L2', 'L3')** in `r5b_selection` and in R9's `names` for `rung2_selected_level`. The tie still goes to the earlier level. R5's B9 fit stays the rung-0 result.
3. **Pre-register both rules and start a fresh run directory.**
   - Add `comparator: "critic 3 + exact-row nuisance at the level's alpha of m(s~)"` and `selection_set: ["L1","L2","L3"]` to `BARS['b23']`.
   - bars.json then changes, so `20261116_075115Z` correctly refuses to resume. Start a new run directory with R1 (about 30 s).
   - Before that: R0 smoke rerun (about 9 s) and two new B19 tests: the comparator uses the level's α, and B9 is never selected at rung 2.
   - Note on the cache: the resume hash includes the code digest of `rlsetup/critic/*.py` and `scripts/critic_c2.py`, so any edit to those files after R2 starts throws away every cached fit. Everything marked "afterwards" or "report-only" above must therefore go in a separate report script outside `CODE_FILES` that reads the saved npz/json files.

Nothing else needs to change. The text amendments to B1, E21.10 and B23 are recorded in this file and need no code.

## Approved to run (after points 1–3, once QA has signed off)

- **Budget:** the projection is upper 3.22 h, or 3.69 h with R3c and R5b, and typical 0.90–1.03 h, on 1 thread. That is inside the 4 h budget and well inside the 6 h hard stop. Points 1–3 add under a minute, so no repeat cut. The projection does not include:
  - the B16 rows-S fallback (`--runs R5,R6,R7 --rows S`, then R9; small fits, about 10–20 min);
  - a full rerun if S22a fires.

  The machine is shared, so the sim-runner should watch cumulative `timing.json` against 4 h.
- **How the sim-runner runs it:** two invocations, in a new run directory.
  1. Run `R1,R2,R3,R3b,R3c,R3d`. Then stop and check `metrics/R3d.json`:
     - if `S22a.drop_quali` is true, do not continue. Escalate to the lead: a `--drop-quali` rerun is a new full run that may exceed 4 h;
     - if `S22b.escalate_TT_reference_decision` is true, report it to the lead and continue;
     - report the R3b/R3c outcome either way: adopted or triggered, delivered variant, modal c.
  2. Run `R4,R5,R5b,R6,R7,R8,R9`.
     - If R9 raises "pending: R5 --rows S", run `--runs R5,R6,R7 --rows S`, then `--runs R9`.
     - If R9 raises `implementation_fault`, stop and report. Do not work around it.
- **Approved without change:** S22a/S22b and R8 stay on plain critic 3. The R6 ablation stays on the rung-0 critic 1. Critic 2 rung 2 is not scheduled.

## Risks to watch (metric that reveals each)

| Risk | Metric |
|---|---|
| Critic-3 fits stall, or the twins diverge | `R3.json: stalled_fits`; twin-0 block vs twin-mean block |
| B21 picks c > 0 on noise | `R3b selection.per_fold` (expected all or mostly 0); per-c Barcelona Δ |
| Rung 2 hides a damaging style path | `R5b selected_counts`, with all L3 meaning the path is effectively off; ΔR² trend L1 → L3 against the matched comparator; the vs-α=10 gap (shows how much credit was the nuisance) |
| OCD/AC styles fall outside the game cells | `e_clipped_frac_test_mean`; R9 `barcelona_style_reference` (clipped fraction, Mahalanobis distance) |
| Pooled m used where it should not be | `pooled_m` flags in R3b and in the R9 bundle |
| Date drift credited to setup | same-month vs all-pairs accuracy (report script) |
| Wall-time overrun on the shared machine | cumulative `timing.json` vs 4 h |

## Questions for Ibrahim

None new. Addendum B21–B23 "Needs Ibrahim" 1–3 (definition of "unsuccessful in grading", whether Π stays on in OCD/AC queries, target session) are still open. None blocks R2–R9. If his answer to 1 changes the trigger, apply it afterwards from the saved fits in a separate script; do not edit the run code mid-run.

Files: `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/scripts/critic_c2.py` (`run_R5b`, `r5b_selection`, `metrics_R5b`, `run_R9`), `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/c2_config.py` (`BARS['b23']`), `/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent/rlsetup/critic/train.py` (`finetune`, `refit_nuisance_ft`).
