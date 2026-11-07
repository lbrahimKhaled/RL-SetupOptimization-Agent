# Plateau fixes for the SAC driver: QSS-relative corner reward, focused starts, network resets

Prepared by `web-researcher`, 7 Nov 2026. Read-only on code.

**Context (from the caller's run logs, 12.35M env steps):**
- Best clean deterministic lap: 102.42 s at 2.5M steps, 101.87 s at 12.35M.
- Deterministic eval laps clean in 50 of 248 evals (about 20 %). Stochastic laps about 60 % clean, mean about 104.5 s. Minimum edge margin 0.10 m.
- Off-tracks cluster at s ≈ 2350-2500 m (53), 650-800 m (36), then about 1550, 4700 and 5100 m.
- Entropy about -2.9 (target -3), alpha about 0.05. The critic overestimates the 1-step MC return by about 2-6 %.
- QSS lap 93.83 s.

**Tags.**
- **[paper]**: what the cited source says (paper, repo or docs).
- **[repo]**: what our own code says.
- **[J]**: my judgment or derivation.
- **NOT OPENED**: I could not read the source, and nothing is filled in from memory.

This note builds on `docs/research/driving_agent_sota.md` (25 Oct) and `docs/research/reward_redesign.md` (26 Oct).

**Code facts checked today (read-only) [repo]:**
- `rlsetup/driving/env.py` lines 477-482 apply F = γΦ(s') - Φ(s) per decision, with Φ = -k|e|, and set Φ(s') = 0 after a terminal failure.
  - So a crash at lateral offset e pays a shaping of +k|e| on the terminal step: a few units, against P = 100.
  - This is required for PBRS (Grześ 2017: Φ(terminal) = 0). It is worth knowing, not a bug.
- `rlsetup/driving/relabel.py` already relabels stored 1-step rewards in place. Its docstring: "SB3's NStepReplayBuffer builds n-step returns at sampling time from these 1-step fields, so the n-step targets follow the relabelled rewards and flags."
  - So re-weighting a stored bonus component at sample time is compatible with our n-step setup (see A.5).

---

## 0. Sources actually opened (this note)

| Key | Source | How accessed |
|---|---|---|
| TAL23 | Evans, Engelbrecht, Jordaan, "High-speed Autonomous Racing using Trajectory-aided Deep RL", arXiv:2306.07003 | arXiv HTML |
| EVANS21 | Evans, Engelbrecht, Jordaan, "Reward Signal Design for Autonomous Racing", arXiv:2103.10098 | abstract page only |
| TRAD26 | Leng et al., "Expert Knowledge-driven RL for Autonomous Racing via Trajectory Guidance and Dynamics Constraints", arXiv:2603.05842v3 | arXiv HTML (first 100k of 117k chars) |
| TRI24 | Subosits et al., "Towards an Autonomous Test Driver: High-Performance Driver Modeling via RL", arXiv:2412.03803 | arXiv HTML |
| AARK24 | Bockman, Howe, Orenstein, Dayoub, "AARK: An Open Toolkit for Autonomous Racing Research", arXiv:2410.00358v2 | arXiv HTML |
| JARITZ18 | Jaritz, de Charette, Toromanoff, Perot, Nashashibi, "End-to-End Race Driving with Deep RL", ICRA 2018, arXiv:1807.02371 | ar5iv full text |
| BARI25v | Bári, Palkovics, "Vision based driving agent for race car simulation environments", arXiv:2504.10266v1 | arXiv HTML |
| ZHAO26 | Zhao, Zhao, Wu, Lv, "Track-Guided Hierarchical RL for Autonomous Vehicle Drifting with Minimum-Lap-Time Planning", arXiv:2608.00113 | abstract page only |
| FUCHS21 | Fuchs et al., "Super-Human Performance in Gran Turismo Sport Using Deep RL", RA-L 2021, arXiv:2008.07971v2 | arXiv HTML |
| LEE25 | Lee et al., "A Champion-level Vision-based RL Agent for Competitive Racing in GT7", RA-L 2025, arXiv:2504.09021v1 | arXiv HTML |
| SONG23 | Song, Romero, Müller, Koltun, Scaramuzza, "Reaching the Limit in Autonomous Racing: Optimal Control versus RL", Science Robotics 8(82), 2023, doi:10.1126/scirobotics.adg1462, arXiv:2310.10943v2 | arXiv HTML |
| SOPHY22 | Wurman et al., Nature 602 (2022), doi:10.1038/s41586-021-04357-7 | **NOT OPENED today**: Nature redirects to login, and the author PDF downloaded but could not be text-extracted here. Statements come only via LEE25 and our 25 Oct note. |
| L2R21 | Herman et al., "Learn-to-Race", ICCV 2021, pp. 9793-9802 | **reward NOT verified**: the repo tree did not expose the reward file, and the docs URL returned 404 |
| NG99 | Ng, Harada, Russell, "Policy invariance under reward transformations", ICML 1999, pp. 278-287 | via the 25 Oct note (abstract) and as restated in DK12 Sec. 2.2 |
| DK12 | Devlin, Kudenko, "Dynamic Potential-Based Reward Shaping", AAMAS 2012, pp. 433-440 | full text (White Rose eprint 75121) |
| WIE03 | Wiewiora, Cottrell, Elkan, "Principled Methods for Advising RL Agents", ICML 2003, pp. 792-799 | **abstract/metadata only**: the PDF fetch returned empty |
| WIE03J | Wiewiora, "Potential-based shaping and Q-value initialization are equivalent", JAIR 19:205-208, 2003 | **NOT OPENED**; result as restated in DK12 Sec. 2.2 and 3.2 |
| HAR15 | Harutyunyan, Devlin, Vrancx, Nowé, "Expressing Arbitrary Reward Functions as Potential-Based Advice", AAAI 2015, doi:10.1609/aaai.v29i1.9628 | search abstract only |
| KNOX | Knox et al., "Reward (Mis)design for Autonomous Driving", arXiv:2104.13906v2 | via the 26 Oct note |
| FREI24 | Freitag, Ceder, Laezza, Åkesson, Chehreghani, "Sample-Efficient Curriculum RL for Complex Reward Functions", arXiv:2410.16790v1 (submitted to AAMAS 2025) | arXiv HTML |
| TRACEY23 | Tracey et al., "Towards practical RL for tokamak magnetic control", arXiv:2307.11546v2 | arXiv HTML |
| D2S20 | Luo et al., "Dense2Sparse Reward Shaping for Robot Manipulation with Environment Uncertainty", arXiv:2003.02740 | abstract only |
| OVERCOOKED-type anneals | arXiv:2605.24423 and arXiv:2302.01605 | search snippets only (linear decay of the shaping coefficient to 0) |
| WOL24 | Wołczyk et al., "Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem", ICML 2024, arXiv:2402.02868 | abstract only |
| FLOR17 | Florensa, Held, Wulfmeier, Zhang, Abbeel, "Reverse Curriculum Generation for RL", CoRL 2017, arXiv:1707.05300 | ar5iv full text |
| TAV18 | Tavakoli, Levdik, Islam, Smith, Kormushev, "Exploring Restart Distributions", RLDM 2019, arXiv:1811.11298 | ar5iv full text |
| SAL18 | Salimans, Chen, "Learning Montezuma's Revenge from a Single Demonstration", NeurIPS 2018 Deep RL workshop, arXiv:1812.03381 | ar5iv full text |
| PENG18 | Peng, Abbeel, Levine, van de Panne, "DeepMimic", ACM TOG 2018, arXiv:1804.02717 | ar5iv (first 100k chars) |
| GOEX | Ecoffet, Huizinga, Lehman, Stanley, Clune, "Go-Explore", arXiv:1901.10995v4 | abstract only |
| KL02 | Kakade, Langford, "Approximately Optimal Approximate RL", ICML 2002, pp. 267-274 | **NOT OPENED**; abstract-level via search results |
| AKLM21 | Agarwal, Kakade, Lee, Mahajan, "On the Theory of Policy Gradient Methods", JMLR 22 (2021) | search snippets only |
| NIK22 | Nikishin, Schwarzer, D'Oro, Bacon, Courville, "The Primacy Bias in Deep RL", ICML 2022, arXiv:2205.07802 | ar5iv full text |
| DORO23 | D'Oro, Schwarzer, Nikishin, Bacon, Bellemare, Courville, "Sample-Efficient RL by Breaking the Replay Ratio Barrier", ICLR 2023, OpenReview OpC-9aBBVJe | **abstract only**: the OpenReview PDF sits behind a bot check. SR-SAC reset schedule NOT verified. |
| SOKAR23 | Sokar, Agarwal, Castro, Evci, "The Dormant Neuron Phenomenon in Deep RL" (ReDo), ICML 2023, arXiv:2302.12902 | ar5iv full text |
| LYLE23 | Lyle et al., "Understanding Plasticity in Neural Networks", ICML 2023, arXiv:2303.01486 | ar5iv full text |
| LYLE24 | Lyle et al., "Disentangling the Causes of Plasticity Loss in Neural Networks", arXiv:2402.18762 | abstract |
| ASH20 | Ash, Adams, "On Warm-Starting Neural Network Training", NeurIPS 2020, arXiv:1910.08475 | abstract only. The shrink-and-perturb formula was **not verified** from the paper. |
| BBF23 | Schwarzer et al., "Bigger, Better, Faster" (BBF), ICML 2023, arXiv:2305.19452 | ar5iv full text |
| NAU24a | Nauman, Bortkiewicz, Miłoś, Trzciński, Ostaszewski, Cygan, "Overestimation, Overfitting, and Plasticity in Actor-Critic: the Bitter Lesson of RL", ICML 2024, arXiv:2403.00514 | ar5iv |
| BRO24 | Nauman et al., "Bigger, Regularized, Optimistic" (BRO), NeurIPS 2024, arXiv:2405.16158 | ar5iv (first 100k chars) |
| INJ23 | Nikishin et al., "Deep RL with Plasticity Injection", NeurIPS 2023, arXiv:2305.15555 | ar5iv full text |
| RDE23 | Kim, Shin, Park, Sung, "Sample-Efficient and Safe Deep RL via Reset Deep Ensemble Agents", NeurIPS 2023, arXiv:2310.20287 | abstract and slides via search |
| ALTNET25 | Maheshwari, Raisbeck, Castro da Silva, "AltNet: Addressing the Plasticity-Stability Dilemma in RL", arXiv:2512.01034v2 (AAMAS 2026 proceedings per search) | arXiv HTML |
| HT24 | Lee et al., "Slow and Steady Wins the Race: Maintaining Plasticity with Hare and Tortoise Networks", ICML 2024, arXiv:2406.02596 | abstract only |
| MA24 | Ma et al., "Revisiting Plasticity in Visual RL: Data, Modules and Training Stages", ICLR 2024, arXiv:2310.07418 | abstract |
| GRAMA25 | Liu, Wu, Obando-Ceron, Castro, Courville, Pan, "Measure gradients, not activations!", arXiv:2505.24061v2 | abstract |
| KUMAR21 | Kumar, Agarwal, Ghosh, Levine, "Implicit Under-Parameterization", ICLR 2021, arXiv:2010.14498 | abstract |
| TQC20 | Kuznetsov, Shvechikov, Grishin, Vetrov, "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics", arXiv:2005.04269 (ICML 2020) | abstract |

---

## A. Corner-specific QSS-relative reward (Ibrahim's idea)

### A.1 Prior work on reference-speed and speed-profile rewards

| Work | What the reward does with a reference speed | Finding | Setting |
|---|---|---|---|
| **TAL23** [paper] | r_TAL = 1 - \|v_agent - v_classic\| - \|δ_agent - δ_classic\| (Eq. 3, Sec. III-C), scaled by 0.2 and clipped ≥ 0. v_classic is the speed of the upcoming waypoint on a min-curvature line with a min-time speed profile (Heilmeier et al.). Crash -1, lap +1. | At 6 m/s the baseline "completes less than 25% of the laps" on ESP/GBR/MCO, and TAL "over 75%" (Sec. IV-C). Sec. IV-B gives 60 % on ESP, which contradicts IV-C. But "the baseline agent achieves slightly lower lap times than the TAL agent" (Sec. IV-C). TAL "approximately tracks the classical planner", "occasionally deviating by selecting higher speeds" (Sec. IV-D/E). | F1TENTH sim, 10 Hz planning, 100 Hz dynamics, v_max 4-8 m/s |
| **EVANS21** [paper, abstract only] | compares position-, velocity- and action-based rewards, including one relative to a min-curvature trajectory | the reward based on distance and velocity relative to a min-curvature trajectory gave the fastest laps | F1/10 sim |
| **TRAD26** [paper] | r_TS = (u_tar - min(\|u - u_tar\|, u_tar)) / max(u_tar, ε) (Eq. 18, Sec. 3.2.3), weight 0.6. u_tar is the reference speed at a look-ahead point. The profile comes from "GGV-envelope-based forward-backward integration" (Sec. 3.2.1), i.e. a QSS profile like ours. | Two-stage curriculum: the speed reference is **removed** at 200k of 250k steps "to overcome the conservatism inherent in the prior trajectory" (Sec. 3.4, 4.1). No ablation isolates the speed term; Table 4 removes only whole modules. The paper does not report whether the agent ended up faster or slower than the reference profile. | unnamed simplified sim, linear tyres |
| **TRI24** [paper] | imitation term R_imi = -(v(u) - μ(u))²/max(σ,σ_min)² (Eq. 6). Here v(u) is the **lateral** offset, not speed. | slightly faster with the term: 96.42 vs 96.69 s (Table II). The authors call the differences small relative to seed variation. | Dymola GT3, 10 Hz |
| **BARI25v** [paper] | "time-difference" reward against a reference lap (Eq. 1). The reference is "a constant speed movement along the track centerline", so the reward "simplifies to the progress made in each time step". | first lap at about 10M steps; about 1000M steps to "professional human level" (Fig. 4) | TORCS, 20 Hz |
| **ZHAO26** [paper, abstract only] | an instant term plus an end term "derived from the Minimum-Lap-Time objective"; the MLT trajectory is used as prior data | only qualitative results on the abstract page | drifting sim |
| **JARITZ18** [paper] | R = v(cos α - d) (Eq. 1), with margin and sigmoid variants on d; no reference speed | distance-penalised rewards cut hits per km from 9.2 to 2.3, but were slower (91.4 vs 106.3 km/h) than Mnih's angle/speed reward (Sec. IV-C2) | WRC6 rally game |
| **FUCHS21, SOPHY22, LINESIGHT, ACGYM24** [paper, via the 25-26 Oct notes] | no reference-speed term. They use progress (plus a time cost), off-course and wall penalties (SOPHY and FUCHS scale these with v²); ACGYM uses v·(1-\|gap\|/12). | the fastest published agents (super-human GT, Trackmania records) use no speed target | GT Sport, Trackmania, Assetto Corsa |
| **SONG23** [paper] | progress-to-gate reward, no reference speed | "RL outperforms OC because it optimizes a better objective" (Results). In the nominal sim, tracking the time-optimal trajectory gave the best mean lap (4.92 s) but only 44 % success; RL got 5.14 s with 100 % success. With a realistic model, trajectory tracking had 0 % success. | drone racing |
| **AARK24** [paper] | the speed reward is capped early. After a full replay buffer of successful episodes, "reward for pace is increased each time a successful episode is completed" until "effectively uncapped" (Sec. VII-A). | without the cap, agents "become stuck in a local reward maxima, where they fail to link a full lap together"; stability "was greatly improved" | Assetto Corsa, SAC |
| **L2R21** | reward not verified | — | Arrival sim |

**Corner- or segment-specific bonuses.** I found none in racing RL (two web searches, plus the papers above). The closest is SOPHY22's doubled off-course penalty in the Sarthe chicanes (per the 25 Oct note), and that is a penalty, not a speed bonus.

**Summary of the evidence [J].**
- Speed-reference rewards help agents *learn to complete laps* (TAL23, TRAD26).
- The two papers that report on speed found the reference-speed agent **slightly slower** (TAL23), or found the reference made it **conservative**, so it was removed late in training (TRAD26).
- The fastest agents use no speed reference.
- SONG23 is direct evidence that tracking a time-optimal (QSS-like) reference is fast in nominal conditions but fragile under model mismatch. That is our situation if v_qss is optimistic.

### A.2 Theory: can the term be potential-based, and what is lost

**What the sources say**
- NG99 [paper]: F(s,s') = γΦ(s') - Φ(s) leaves the optimal policy unchanged, and the potential-based form is also necessary for invariance (abstract, via the 25 Oct note).
- DK12 [paper]:
  - A time-varying potential keeps invariance if F(s,t,s',t') = γΦ(s',t') - Φ(s,t), with each potential evaluated at the time its state was visited (Eq. 4, Sec. 3.1). The shaped Q* equals Q* - Φ(s,t) (Eq. 8).
  - Static PBRS is equivalent to initialising the Q-table as Q(s,a) = Φ(s) (WIE03J, restated in DK12 Eq. 9). Dynamic PBRS is not (Sec. 3.2).
  - All these proofs "require actions to be selected by an advantage-based policy" (Sec. 2.2).
  - Single-agent demo: a misleading, never-stable potential still converged to the optimum, but learned more slowly (Sec. 4.1).
- WIE03 [paper, abstract only]: extends PBRS to state-action potentials ("advice") without changing the optimal policy, with two conversion methods. I could not open the text to give the exact action-selection condition.
- HAR15 [paper, abstract only]: an arbitrary shaping reward R† can be turned into a dynamic potential by learning an auxiliary value function of R† online. The resulting advice "captures the input reward function in expectation".
- Grześ 2017 (via the 25 Oct note): Φ(terminal) = 0 is needed in episodic tasks.

**My reading [J]**
1. **The term as stated is not potential-based.** A bonus for |v - v_qss| being small inside a corner window changes the optimal policy, by NG99. The new optimum trades lap time against hitting v_qss. That is fine for a training aid, but at convergence it is a bias.
2. **A potential-based version exists.** Two natural potentials on the state x = (s, v, e, ...):
   - *Time-to-go potential*: Φ(x) = -b·T_qss(s, v). T_qss is the QSS time from (s, v) to a fixed point ahead (e.g. the next corner exit), computed with the same forward/backward pass started at speed v.
     - Above the QSS braking envelope, set T large. An equivalent "entry-overspeed" potential is Φ = -P·σ((v - v_env(s))/w), where v_env is the backward-pass braking envelope.
     - This makes the "you are now too fast to make the next corner" signal arrive at the moment of overspeed, not 1-3 s later at the crash.
     - It cannot change the optimum. SAC's soft-optimal policy is also unchanged, because Q_soft shifts by Φ(s), which is a state-only offset.
   - *Speed potential*: Φ(x) = c·min(v, v_qss(s)). It pays for gaining speed up to v_qss and takes the payment back when speed is lost, so overall it only reshapes timing.
   - Multi-lap wrap: Φ must be continuous across the start/finish line. A function of (s mod L, v) is fine; a function of absolute progress is not.
3. **What PBRS loses** is exactly what Ibrahim may want: a *permanent* pull toward v_qss.
   - PBRS cannot make the converged driver faster or safer than the unshaped optimum. It only changes the learning path, and DK12 Sec. 4.1 shows it can slow learning if Φ is misleading.
   - By the Q-initialisation equivalence (WIE03J via DK12), adding a *static* potential after 12M steps is roughly a one-off shift of the critic's targets by -Φ(s). The critic has to re-fit, and then the effect fades.
   - After a long run, I expect PBRS to help little unless the problem is credit assignment over the 1-3 s between overspeed and crash.
     - n-step 3 at 25 Hz covers only 0.12 s.
     - γ = 0.992 gives a 5 s horizon, so the crash is inside the horizon, but its signal is learned through many bootstraps.
4. **The HAR15 route** (learning a potential from an arbitrary corner bonus) keeps invariance but costs a second critic. Not worth it here.

### A.3 Design alternatives and risks

**Risks of the term as stated [J]:**
- **(a) Optimistic v_qss rewards overspeed.** QSS has no transients and no yaw-stability limit, so the true achievable apex speed may be below v_qss.
  - A symmetric "close to v_qss" reward then pulls entry speed up exactly where the deterministic policy already fails (minimum edge margin 0.10 m).
  - SONG23 is the closest empirical warning: tracking a time-optimal reference gave 44 % success nominally and 0 % with a realistic model.
- **(b) Hacking.** A per-tick reward inside a window can be collected, and the car can then crash after the window. With γ = 0.992 per decision, a crash after the window is discounted, so the agent may accept late crashes if the window reward is large.

**Design alternatives [J]:**
- **(c) Clean-corner bonus at exit.** Pay B at corner exit only if the car is still on track, and optionally only if v_exit ≥ (1-x)·v_qss_exit.
  - This removes hack (b) for that corner.
  - It is still non-potential and shifts the optimum toward high exit speed. That is the right direction for lap time, because exit speed carries down the straight, and the bonus is conditioned on survival.
- **(d) Asymmetric shape.** Reward ∝ min(v, v_qss)/v_qss, saturating at v_qss, and zero or negative above v_qss·(1+m).
  - This never rewards overspeed. If v_qss is optimistic, the cap is simply not binding.
  - For contrast, TRAD26's r_TS (Eq. 18) and TAL23's term are both symmetric: being above the target is penalised as much as being below it.
- **(e) Anneal to zero.** TRAD26 drops its term at 80 % of training, and AARK24 used a speed curriculum. Annealing means the final driver is again optimal for the unshaped reward, but only if training continues long enough after the term reaches zero. Details in A.5.

**Magnitude versus P and γ (derived [J]):**
- Per decision (2 ticks, 0.04 s) the base reward is about a·v·0.04 - b·0.04 = 0.04(v - 1). At the lap-average speed of about 5450/102 ≈ 53 m/s, that is about 2.1 per decision.
- With γ = 0.992 the horizon is 1/(1-γ) = 125 decisions (5 s). The value of staying alive is therefore V ≈ 2.1 × 125 ≈ 260.
- A crash costs about **P + V ≈ 100 + 260 ≈ 360** return units, not just P.
- A bonus B makes a riskier line worthwhile when ΔB > Δp_crash·(P + V) - (time gain).
- For the bonus alone never to justify +1 % crash probability per corner, B must stay below about 0.01 × 360 ≈ 3.6 per corner.
- For comparison, gaining 0.1 s in a corner is worth about v × 0.1 ≈ 5 m of progress, i.e. about 5 units.
- So a bonus big enough to matter (worth about 0.05-0.1 s) is already big enough to buy about 1 % more crash risk. That is the core tension: a QSS bonus strong enough to change behaviour is strong enough to bias it.

### A.4 Bias for setup scoring

- **Literature:** I found no paper on how a reference-anchored reward biases a downstream setup ranking. The nearest is TRI24 [paper]: with a lateral-imitation term, its driver ranked four setups in the same order as a pro driver. That term was setup-independent and small (Tables II/III).
- **Argument [J]:**
  - A non-potential term anchored to a **per-setup** v_qss makes each setup's driver optimise "lap time + closeness to that setup's QSS".
  - The QSS error depends on the setup. QSS sees only the steady-state g-g-v, so transient parameters (dampers, yaw-inertia effects, diff behaviour) enter weakly or not at all.
  - Setups with a more optimistic QSS therefore get a driver pushed harder toward overspeed, meaning more crashes or slower recovery. Setups with a pessimistic QSS get a driver held back.
  - The lap-time ranking then drifts toward the QSS ranking. That is exactly the model error the RL driver is supposed to remove.
  - Knox et al. (Sec. 4.1, via the 26 Oct note) warn that shaping changes "the reward function's preference ordering over policies".
- **Safe uses [J]:**
  1. Use the PBRS form, which has no bias at the optimum.
  2. Use a training-only term faded to zero before any lap time is used for scoring, and run the per-setup fine-tune on the unshaped reward. This is Ibrahim's preferred direction; see A.5.
  3. Use a setup-independent reference (e.g. the base car's profile). This avoids a per-setup bias but injects a constant one.

### A.5 Fading the corner bonus to zero (Ibrahim's preferred direction)

Scope: only the **new corner/QSS bonus** is faded. The final reward is plain `time_pbrs`: a·Δs - b·Δt - P·[failure] + γΦ' - Φ with Φ = -k|e|. This is **not** the superseded idea of annealing the line-tracking weight c to 0 (problem_formulation_v5), and I do not recommend that.

#### A.5.1 Prior work that anneals or switches shaping and auxiliary rewards

| Work | What changes | Schedule | Result reported |
|---|---|---|---|
| TRAD26 [paper] | MCRL speed reference removed | **step** at 200k of 250k steps (80 %); fixed threshold (Sec. 3.4, 4.1) | stated aim: overcome conservatism and "surpass the MCRL baseline". **No ablation shows it did.** |
| AARK24 [paper] | speed reward cap lifted | **performance-triggered**: starts after a full buffer of successful episodes, then increases at each successful episode until uncapped (Sec. VII-A) | stability "greatly improved". No retention data. |
| FREI24 [paper] | phase 1 = goal + PBRS + velocity; phase 2 adds collision, action and tracking terms (Eq. 2) | **step** at T_cr = 500k of 1M steps (Sec. 4.4). Suggests watching convergence of the α loss as a switch signal, but does not automate it. | Buffer stores **both** rewards per transition and switches all samples to r_full at T_cr (Alg. 1); n_cr = 50k extra gradient steps at the switch (best of 1k / 25k / 50k, Fig. 5). Success rate: RC-SAC 66 % vs SAC 42 %. The naive curriculum (buffer reset, no extra steps) fell *below* baseline after the switch (Sec. 4.5, Fig. 1). "Reusing past experience is essential" (Fig. 5). |
| TRACEY23 [paper] | reward parameter narrowed from 0.1 to 0.025 over 6M updates; schedule "stationary at approximately 60% through training" (Sec. 3.1.2) | **linear-type** schedule | no significant gain: RMSE 0.490 vs 0.567, while a fixed exacting reward reached 0.201. Hypothesis: the value function "struggles to stay in line with the latest reward function as it evolves" and receives "different learning signals for the same states at different points in training" (Sec. 3.1.2, MPO). On the harder task, a two-stage (forgiving → exacting) reward worked better than either alone (Sec. 3.1.3). |
| D2S20 [paper, abstract] | dense → sparse reward switch | not stated in the abstract | higher expected reward than dense-only or sparse-only, and more robust to uncertainty |
| Overcooked-style anneals [search snippets only] | shaped-reward coefficient decayed linearly to 0 (about 1.5×10⁷ steps in arXiv:2605.24423; 1 → 0 in arXiv:2302.01605) | **linear** | snippets only; no retention numbers seen |
| WOL24 [paper, abstract] | fine-tuning a pre-trained policy | — | the model "deteriorates on the state subspace of the downstream task not visited in the initial phase of fine-tuning"; standard knowledge-retention techniques mitigate this |

**Retention after fade-out: what the evidence actually says.**
- I found **no study that measures whether speed or robustness gained under a shaping term survives after the term is removed** in racing or locomotion (three searches plus the papers above).
- TRAD26 and AARK24 report only end results, not before/after-fade comparisons.
- FREI24 shows that *adding* terms after a phase works with buffer reuse.
- TRACEY23 is the only source that reports a scheduled reward doing **no better** than a fixed one, and it blames critic lag.
- Retention is therefore an **open empirical question for us**. A control branch is mandatory (A.5.4).

#### A.5.2 What happens inside SAC when the reward changes under it [J, grounded where cited]

1. **Stale stored rewards.** If the buffer stores the total reward computed with the old bonus weight, every sampled target uses a mix of old weights.
   - The fix FREI24 uses is to store the components and recombine them at update time.
   - Our case: store r_base (time_pbrs, including P and the shaping F) and r_bonus separately. At sampling, use r = r_base + w(t)·r_bonus.
   - Our `relabel.py` docstring confirms that SB3's NStepReplayBuffer builds n-step returns at sampling time from the 1-step fields, so the n-step targets follow w(t) automatically.
   - Then the buffer is always consistent with the current reward, and at w = 0 it is exactly time_pbrs. **Relabel on the fly; do not keep stale stored totals.**
2. **Critic lag (Q-target staleness).** Even with consistent rewards, Q must re-fit as w falls. Near corners, Q drops by about the discounted bonus (≤ B per corner, B ≲ 3.6 per A.3).
   - The target network moves with τ = 0.005, a time constant of about 200 updates.
   - Value information also has to propagate back over about 125 decisions through n-step 3 bootstraps.
   - So the critic needs a fade spread over many thousands of updates, not a cliff. TRACEY23's hypothesis supports this, and FREI24's 50k extra steps at a switch is a lower bound.
3. **Policy jump at the end of the fade.** As w falls, the soft-optimal action near corners shifts from "hit v_qss" to "minimise time". If the bonus had pulled the policy away from the time optimum, the policy moves back.
   - This is intended: it removes the bias.
   - The risk is overshoot or oscillation if the actor follows a critic that has not re-fit yet. A low actor lr during and after the fade (currently 1e-4) limits this.
4. **Entropy and alpha.** The reward scale changes only slightly (B is small relative to about 2.1 per decision), so auto-α should not move much. Log it anyway.
5. **Forgetting.** Removing the bonus does not remove data, because the buffer keeps the corner states.
   - WOL24's mechanism is forgetting of states that stop being visited. Here the corner states keep being visited every lap, so I expect forgetting to be limited to behaviour that is not lap-time-optimal, which is the point of fading.
   - What can be lost is the *robustness margin* the bonus bought, if that margin costs time under time_pbrs. That is a property of the objective, not of the schedule.

#### A.5.3 What is likely retained [J]

- **Likely retained:** what the bonus helped the agent *discover* that is also better under time_pbrs, such as a faster but clean entry it would not have found by itself. That is the same mechanism as PBRS: exploration help.
- **Likely not retained:** what was only held in place by the bonus (e.g. matching v_qss where the time optimum is lower or higher).
- So the bonus is worth it only if the plateau is an *exploration* problem at those corners. If the plateau is an *objective* problem, the fade gives the gains back. The objective may simply prefer the fragile line: time_pbrs gives a constant P regardless of how close to the edge the car runs, and SAC optimises the stochastic policy, not the deterministic one.

#### A.5.4 Concrete schedule for our case (resume from 12.35M steps, 1M buffer)

Phases count env steps from the checkpoint. The numbers are my proposal [J], sized from FREI24 (50k extra updates at a switch, buffer reuse), TRAD26 (last 20 % without the reference) and the critic-lag argument above.

0. **Pre-check (cheap).** From the existing eval logs, compare v_RL(s)/v_qss(s) on the clean laps at the crash windows (650-800, 2350-2500 m). If RL is already *at or above* v_qss there, a speed bonus is the wrong tool; prefer B (starts) or E/F in section D.
1. **Buffer.** Add an `r_bonus` field to the replay buffer, recombining at sample time as r = r_base + w(t)·r_bonus.
   - Old transitions (from before the bonus) get r_bonus computed offline from stored obs/next_obs. The caller says this is possible; the corner-exit test needs s, v and on-track at the exit transition.
   - Alternatively, give old transitions r_bonus = 0 and accept a short inconsistency.
2. **Bonus phase, w = 1, about 1.5-2M steps.** Use an asymmetric clean-exit bonus: B ≤ 3 per corner, paid only at exit if on track, scaled by min(v_exit, v_qss_exit)/v_qss_exit above a floor. Only at the 2-5 crash corners, or at all corners (Ibrahim's choice).
3. **Fade, about 2M steps.**
   - Default: linear w: 1 → 0 (smooth, critic-friendly).
   - Alternative, performance-triggered (AARK24-like): start the fade when the deterministic-eval clean rate at the targeted corners reaches ≥ 80 % over the last 10 evals. Step w down by 0.25 only after each step has held for ≥ 300k env steps without a drop in clean rate.
4. **Bonus-free consolidation, w = 0, ≥ 3M steps** (about 25 % of the run so far; TRAD26 used 20 %).
   - Lap times for any comparison or setup scoring come **only** from this phase, after the last-1M-steps curves of clean rate and median clean lap are flat.
5. **Control branch (mandatory).** From the same 12.35M checkpoint, run time_pbrs only for the same total steps (about 7M) with the same seeds and starts.
   - Without it, any gain could just be more training.
   - Decision rule: keep the bonus recipe only if, in phase 4, the bonus branch beats control on deterministic clean rate **and** is not slower on median clean lap.
6. **Setup fine-tunes later:** time_pbrs only, no bonus (A.4).

---

## B. Concentrating training starts before the crash corners

### B.1 What the sources say

**Start-state curricula and restart distributions:**
- **FLOR17** [paper]:
  - Trains from "good starts" S = {s0 : R_min < R(π, s0) < R_max}, with R_min = 0.1 and R_max = 0.9 (Sec. 4.1, App. A.1).
  - New starts come from short Brownian-motion rollouts: T_B = 50, M = 10,000 collected, N_new = 200 sampled.
  - Each iteration appends N_old = 100 old good starts, "an important feature to avoid catastrophic forgetting" (Sec. 4.3). That is about 1/3 old starts.
  - Training on ρ_i is acceptable because an optimal policy under ρ_i is optimal under ρ0 "if their supports coincide" (Sec. 3.1).
- **TAV18** [paper]:
  - Restarts from states in an experience memory, in three variants: uniform, prioritised by |TD error| (α = 0.4), and episodic (by return) (Sec. 4).
  - Fixed ratio: "10% of the total interactions stem from augmented initial states" (Sec. 5).
  - Caveat: "altering the distribution of initial states could change the optimal policies when searching within a restricted class of policies" (abstract, Sec. 3.1). Keeping the original start support reduces this bias.
  - Results: mild gains on HalfCheetah, significant on Humanoid. On Thrower the baseline failed in 80 % of 42 seeds, while episodic restart always learned (Sec. 5).
- **SAL18** [paper]:
  - Starts from demonstration states. Workers start uniformly in a window {τ-D, …, τ}, and τ moves back when the success proportion reaches ρ = 20 % (Alg. 2, Sec. 4.2).
  - Needs a resettable simulator, and the agent must generalise to near-demo states (Sec. 4.2).
- **PENG18** [paper]:
  - Reference State Initialization (RSI) samples starts along the reference motion (Sec. 6.1).
  - Without RSI the backflip policy "never learns" the flip (Sec. 10.4). Normalised return was 0.791 with RSI+ET vs 0.379 in the ablation column (Table 5; the text's column mapping is slightly ambiguous).
- **GOEX** [paper, abstract]: "first return to a promising state (without exploration), then explore from it"; it relies on the determinism of simulators.
- **KL02 / AKLM21** [paper, not opened / snippets only]: policy-optimisation guarantees depend on the restart distribution μ through a distribution-mismatch coefficient ‖d^π*/μ‖∞. A μ that covers the states the optimal policy visits makes first-order methods converge faster. I did not open the papers to quote the theorem.

**Racing practice [paper]:**
- FUCHS21: positions "equally distributed over the racing track with an initial speed of 100 km/h" (App. VI-C). No curriculum: "warm-starting from imitation … does not improve final performance" (Sec. V-A).
- TRI24: "random position on the track", heading aligned with the centreline, speed "half of what it would be on a full speed lap".
- LEE25 (GT7): "starts each race from randomly sampled points around the track", following GT Sophy's multi-scenario training (Sec. IV-B). Proportions are not given.
- JARITZ18: random checkpoint respawn at 0 km/h gave "significantly better exploration" (+32 % / +65 %) than always starting at the track start.
- SONG23 (drone racing): starts near path-segment centres early, then from an initial-state buffer of states visited by successful runs, which "greatly improves sample efficiency". This is from Materials and Methods as summarised by the fetch tool; I did not copy the exact sentence.
- I found **no racing paper that concentrates starts at the corners where the agent crashes** (two searches). Uniform-over-track starts are the norm.

### B.2 Recommendation for our case [J]

We already start uniformly in s, at a fraction of v_qss. The failures are concentrated: 53 + 36 of the logged crashes fall in two windows. Proposal:

1. **Mixture.** Keep ≥ 70-80 % uniform starts and make 20-30 % "focused" starts.
   - Uniform starts keep the support on the whole track. That is FLOR17's and TAV18's condition for no bias at the optimum, and it protects the other corners from forgetting.
   - TAV18 used 10 % augmented starts and FLOR17 about 1/3 old to 2/3 new; 20-30 % sits between them.
2. **Where.** Use an **adaptive histogram** instead of fixed windows.
   - Bin s in 50 m bins and keep a running count of off-tracks per bin over the last N episodes.
   - Sample the focused starts from bins 100-300 m *before* each failure location, with probability ∝ failure count (plus a floor).
   - This is FLOR17's "intermediate difficulty" idea in 1-D, and it moves automatically when the crash site moves.
3. **Which state.** A start "before the corner" at a fraction of v_qss is not the state the policy is in when it fails: it fails at high speed, on its own line.
   - Better (Go-Explore / SONG23 / TAV18 style): **restart from states the current policy actually visited** at s ∈ [s_fail - 300, s_fail - 100] m, taken from recent rollouts.
   - Store full simulator snapshots if OCD's backend can restore a state. If not, store (s, e, heading, v, yaw rate) and re-initialise the car from those values.
   - Include deterministic-policy states. The deterministic policy is the one that fails at evaluation (80 % failures), and its states are under-represented in the buffer because data is collected with the stochastic policy.
4. **Risks and guards:**
   - Forgetting other corners: keep the uniform starts, and monitor clean rate per corner, not just per lap.
   - Critic distribution shift: off-policy SAC tolerates it, but the replay mix changes. Cap the focused fraction so focused transitions stay below about 30 % of the buffer.
   - Short focused episodes end in crashes more often, so terminal transitions become more frequent in the buffer. That is the point (more crash-corner data), but watch α and the Q scale.
   - Evaluation must stay on the original start distribution: full laps from a fixed start.

**Evidence strength [J]:** medium that restart distributions help in hard-exploration and precision states (FLOR17, TAV18, SAL18, PENG18); low for racing specifically, since I found no corner-focused racing example.

---

## C. Network resets and plasticity loss

### C.1 What each method resets, how often, and the cost

| Method | What is reset | How often | Replay ratio (RR) | Temporary drop / notes |
|---|---|---|---|---|
| **NIK22** primacy bias [paper] | SAC: **all** networks (actor, both Q, targets) and the optimizer state; buffer kept (Sec. 5.1) | every 2×10⁵ env steps | 1-256 (Table 5a) | "spikes of reduced performance"; on cheetah-run it returns "in just a few thousands of steps" (Sec. 5.3). IQM resets vs no resets: **RR 1: 401 vs 500** (resets hurt), RR 9: 628 vs 459, RR 32: 651 vs 301, RR 128: 584 vs 149. DrQ ablation (App. B, Fig. 11): resetting the **critic alone had the largest effect in most environments**, and resetting all was most robust. Resetting the full 3-layer head was slightly better than the last layer only (Fig. 10). Resetting the buffer too ≈ learning from scratch. |
| **DORO23** SR-SAC [paper, abstract only] | full or partial resets | schedule **NOT verified** (OpenReview PDF blocked) | scales RR upward | abstract: resets enable better scaling with the replay ratio on DMC and Atari 100k |
| **BBF23** [paper] | conv layers moved "50% towards the random target" (shrink-and-perturb style), harder than SR-SPR's 20 % | every 40k gradient steps, "following the schedule of D'Oro et al." (Atari) | 8 | resets "essential" (Fig. 5) |
| **SOKAR23** ReDo [paper] | only τ-dormant neurons: incoming weights re-initialised, outgoing weights set to 0 (Alg. 1) | SAC: every 200k steps | various | **no performance drop** by construction (output unchanged at τ = 0). In MuJoCo SAC, ReDo was the only method without degradation, while reset and weight decay caused drops in most cases. No gain at full width; helped on Ant at half and quarter width (Table 6). "Strong correlation between replay ratio and the fraction of neurons turning dormant" (Sec. 3). |
| **INJ23** plasticity injection [paper] | head (last 2 layers): freeze θ, add trainable θ'₁ minus a frozen copy θ'₂, so the output is **unchanged at injection** (Sec. 4) | once (50M of 200M frames) | Atari | no reset drop. Also a **diagnostic**: inject at a plateau and compare; if it improves, plasticity was the limit; if not, the cause is e.g. exploration (Sec. 5.2). Effect grows with higher lr and RR and with smaller nets. |
| **NAU24a** [paper] | "full-parameter resets" (actor-critic; which networks is ambiguous on the page) | SAC period not stated on the page (DrQ: every 200k steps) | 2 and 16 | **LayerNorm + resets** strongest on DMC at high RR. Overestimation showed the strongest correlation with returns; LN and spectral norm cut overestimation more than critic-specific fixes. Dormant-neuron share correlated negatively with performance on DMC. |
| **BRO24** [paper] | "all parameters" | at 15k, 50k, 250k, 500k, 750k steps (App. D) | 10 (BRO), 2 (BRO Fast) | LN residual critic, about 5M parameters, AdamW. "The largest model without resets nearly matches the performance of the BRO with resets" (Sec. 3). |
| **RDE23** [paper, abstract/slides] | ensemble of N agents, reset one at a time | sequential | DMC (SAC) | motivated by "performance collapses after executing the reset"; the ensemble prevents the collapse |
| **ALTNET25** [paper] | two networks; the passive one is reset and trains **off-policy from the shared buffer** while the active one drives; they swap | every 200k gradient updates | 1 and 4 | standard resets collapse "almost immediately" at RR 1, AltNet avoids the drop. DMC AUC at RR 1: AltNet 543, RDE 510, standard resets 467, SAC 430 (Table 1). |
| **HT24** Hare & Tortoise [paper, abstract] | fast net periodically reset *to the slow net's weights* | not stated in the abstract | Atari 100k | keeps prior knowledge, unlike random resets |
| **LYLE23** [paper] | — (analysis) | — | — | LayerNorm was the most effective intervention and "sometimes beat resetting the last layer". Last-layer reset helps. Weight decay and shrink-and-perturb did **not** improve plasticity (App. B.1). Resetting only the optimizer state did not help. |
| **LYLE24** [paper, abstract] | — | — | — | plasticity loss "can be decomposed into multiple independent mechanisms"; **LayerNorm + weight decay** keeps plasticity across tasks, including ALE |
| **MA24** [paper, abstract] | — | — | adaptive RR | "the critic's plasticity loss serves as the principal bottleneck"; loss becomes catastrophic without early intervention |
| **LEE25** (GT7, racing) [paper] | "the networks" (which ones not stated) | **once**, at 2000 epochs, when the buffer reached capacity (Sec. IV-B) | — | "improved stability and final performance" (Sec. V-C), no numbers. **The only racing evidence.** |

### C.2 Is plasticity loss a likely cause of our plateau? [J]

**Points against it being the main cause:**
- We already have a **LayerNorm critic**, which LYLE23, LYLE24 and NAU24a all find to be the strongest single mitigation.
- Our replay ratio is about 2 updates per env step. In NIK22 Table 5a, resets *hurt* at RR 1 and helped only at RR ≥ 9.
  - Please confirm whether "2 gradient steps per env step" is per transition or per vector step of 4 envs. That makes RR about 2 or about 0.5, and 25M or 6M total updates.
- The symptoms point at robustness and exploration more than at a critic that cannot fit:
  - the deterministic policy is faster but crashes 80 % of the time, while the stochastic policy is 60 % clean;
  - crashes concentrate at 2-3 corners;
  - the gain from 2.5M to 12.35M steps was small but steady.
- NAU24a found overestimation the strongest predictor of returns, and ours is 2-6 %.

**Points for it:** 12M steps is a long run, the 3×256 MLPs are small for the data volume, and LEE25 found a single reset helped in a racing task.

**Conclusion:** plausible but not the leading suspect. **Test before acting.**

### C.3 Diagnostics: plasticity loss vs other plateau causes

1. **Decisive A/B (INJ23 Sec. 5.2 logic).** From the 12.35M checkpoint, run two branches for about 1M steps: (i) unchanged, (ii) plasticity injection on the critic head, or a critic re-init with offline re-fit (C.4).
   - If (ii) improves the deterministic clean rate or lap time, plasticity was a limit.
   - If not, look at exploration, the objective or overestimation.
2. **Fresh-critic probe [J, adapted from LYLE23's probe-task method].** With the actor frozen, fit a freshly initialised critic on the 1M buffer for K updates. Compare its TD loss and its MC-return bias to the old critic trained for the same K updates. A fresh net that fits clearly better or faster indicates lost plasticity.
3. **Supporting metrics:**
   - τ-dormant fraction per layer for actor and critic (SOKAR23: τ = 0.025-0.1; SAC analysis at τ = 0).
   - With LayerNorm, prefer the gradient-based GraMa metric. GRAMA25 says the activation-based ratio "loses statistical power in more complex architectures".
   - Effective rank of the critic's penultimate features (KUMAR21: rank drops with bootstrapping).
   - Gradient-norm and weight-norm trends.
   - **Caveat:** LYLE23 found weight norm, feature rank and dormant units "not reliably predictive" across 128 runs (Sec. 5.2). Treat them as supporting evidence only.
4. **Rule out the other causes in parallel:**
   - Deterministic-vs-stochastic gap: evaluate a low-noise policy, e.g. actions at 0.3σ.
   - Overestimation: watch the 2-6 % bias trend.
   - lr: GT agents used 2.5e-5 to 5e-5 (SOPHY22, LEE25), and Linesight decays 1e-3 → 5e-5 → 1e-5 (26 Oct note), while our critic lr is 3e-4.
   - Metric noise: "best clean lap" is a maximum over noisy draws. Track the median clean deterministic lap together with the clean rate.

### C.4 Recommendation for our case [J]

**Do not do full (actor + critic) resets.**
- The actor holds the BC init plus 12M steps of learning.
- The 1M buffer is only about 11 h of driving at 25 Hz (1M / 25 / 3600), roughly 400 laps of recent behaviour.
- RDE23 and ALTNET25 exist precisely because full resets collapse performance. WOL24 shows forgetting in under-visited states.

**If the C.3 test is positive, do a critic-only reset with an offline re-fit (ALTNET25's "passive net trained from the shared buffer" idea, without the twin):**
1. Load the checkpoint and the 1M buffer. Freeze the actor and α.
2. Re-initialise the critic, the target critic and the critic optimizer. Reset the full critic, or at least the full head: NIK22 found the full head slightly better than the last layer, and the critic alone the most effective part in most environments.
3. Train the critic offline on the buffer with SAC targets under the frozen actor.
   - Run about 200k-500k gradient steps. That is the order of ALTNET25's 200k swap interval and well above FREI24's 50k.
   - Continue until its TD loss and its MC-return bias match or beat the old critic.
   - Meanwhile, keep collecting a little fresh data with the frozen actor (WSRL-style warm-up, per the 25 Oct note).
4. Unfreeze the actor with a **lower lr** (e.g. 1e-4 → 3e-5). Keep a rollback checkpoint, and accept only if the deterministic clean rate and the median clean lap are not worse after about 1M steps.

**Cheaper alternatives with no drop:**
- Plasticity injection on the critic head (INJ23).
- ReDo on the critic every 200k steps (SOKAR23).
- Add weight decay to the LayerNorm critic (LYLE24).

**Priority:** after B and E/F in section D, unless the C.3 test is positive.

---

## D. Ranking

Options:
- **A1**: QSS closeness bonus as stated (symmetric, per tick).
- **A2**: asymmetric clean-exit bonus, faded to zero per A.5.
- **A3**: PBRS QSS potential (time-to-go or overspeed).
- **B**: focused starts from visited pre-corner states, 20-30 % mixture.
- **C**: critic reset + offline re-fit, only after a positive diagnostic.
- **E** (alternative): distributional critic (TQC in sb3-contrib; QR-SAC in SOPHY22) plus a lower late lr.
  - Evidence: SOPHY22 Fig. 2a (via the 25 Oct note): without QR, GT Sophy "would not be faster than the best human on Maggiore".
  - TQC20 abstract: truncation gives "arbitrary granular overestimation control".
  - GT agents used critic lr 2.5e-5 to 5e-5.
- **F** (alternative): close the deterministic-vs-stochastic gap. Collect a fraction (e.g. 10-25 %) of rollouts with a low-noise or deterministic policy, so its states are in the buffer, and/or lower the target entropy. This is my diagnosis from the 20 % vs 60 % clean rates; I found no source testing it in racing.

| Option | (i) speed | (ii) clean-lap rate | (iii) bias for setup scoring | (iv) effort | (v) evidence |
|---|---|---|---|---|---|
| A1 symmetric QSS bonus | ? (TAL23: slightly slower) | risk ↓ if v_qss optimistic (SONG23) | **high** (anchors to per-setup QSS) | low | weak / negative |
| A2 asymmetric clean-exit bonus, faded | + if exploration-limited | + during bonus; retention unproven | low after fade (lap times only from the w = 0 phase) | medium (buffer field, schedule, control branch) | weak (TRAD26, AARK24, FREI24; no retention study) |
| A3 PBRS QSS potential | 0 at optimum, faster learning possible | + via earlier overspeed signal (J) | none | medium | theory strong (NG99, DK12); empirical in racing none |
| B focused starts | small + | **+** at the crash corners | none (eval on the original start distribution) | low-medium | medium generic (FLOR17, TAV18, PENG18), none racing-specific |
| C critic reset + re-fit | ? | ? | none | medium | strong in DMC at high RR; weak at our RR (NIK22 RR 1 negative); one racing anecdote (LEE25) |
| E TQC/QR critic + lr decay | + (SOPHY22) | + via less overestimation (J) | none | low (sb3-contrib TQC; needs a new critic) | medium (SOPHY22 ablation, TQC20) |
| F low-noise rollouts / entropy | + (the det policy is already faster) | **+** if the det-vs-stoch gap is the cause | none | low | weak (my diagnosis) |

**Suggested order [J]:**
1. Run the cheap diagnostics: the A.5.4 step-0 speed check, the C.3 A/B, and a low-noise evaluation.
2. Apply **B + F** together (both low effort, no bias).
3. Then **A2**, faded, with a control branch.
4. Use **E** as the next structural change.
5. Use **C** only if the diagnostic is positive.

---

## Implications for us (interpretation)

1. If the corner bonus is used, make it asymmetric (never reward v > v_qss), pay it at clean corner exit, keep B ≲ 3 per corner (crash cost ≈ P + V ≈ 360), store it as a separate buffer field re-weighted at sample time, fade it over about 2M steps, and score only from ≥ 3M bonus-free steps after the fade. Always run a time_pbrs-only control branch from the same checkpoint.
2. Check first whether the RL speed at the crash corners is already at or above v_qss. If it is, a speed bonus cannot help there, and starts (B) or data from the deterministic policy (F) are the better levers.
3. Use focused starts for 20-30 % of episodes, restarted from states the policy actually visited 100-300 m before failure hot spots, adaptively weighted by failure counts. Keep ≥ 70 % uniform starts and evaluate unchanged.
4. Do not reset the actor. Test plasticity with an injection or fresh-critic A/B first. If positive, reset only the critic, re-fit it offline on the 1M buffer with the actor frozen, then unfreeze the actor at a lower lr.
5. Report the median clean deterministic lap and the clean rate, not only the best lap.
