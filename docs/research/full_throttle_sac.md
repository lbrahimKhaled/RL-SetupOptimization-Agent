# Full throttle with SAC: why the driver stays off the bound, and what fixed it elsewhere

Prepared 20 Nov 2026. Question: on Barcelona straights the SAC driver is at full throttle only 0–56 % of the time and touches the brake on 12–18 % of some straights. The lap is about 6 s slower than the QSS optimum, and about 3.6 s of that is on straights. Apex speeds are already above optimum. Hypothesis to test: tanh squashing plus the entropy bonus hold the throttle mean away from +1, and the time/PBRS reward makes the gain from the last few % of throttle small compared with noise.

Our setup, read from the repo on 20 Nov 2026:
- SB3 2.9 SAC, `ent_coef='auto_0.1'`, `target_entropy='auto'` = −3 (`rlsetup/driving/train_config.py:39-40`).
- 3 actions [steer, throttle, brake] in [−1, 1], 25 Hz decisions, γ = 0.992 (`rlsetup/driving/config.py:69`).
- Pedal deadzone mapping `pedal = clip((a + 0.5)/1.5, 0, 1)` (`rlsetup/driving/demo_driver.py:36-42`).
- "Full throttle" in eval means pedal ≥ 0.95 (`rlsetup/driving/train_config.py:126`).
- No gSDE, so the actor has no `clip_mean`.

Convention: **[S]** = what the source says (quoted or closely paraphrased). **[R]** = my reading, derivation or recommendation.

---

## 0. Bottom line

1. **[R] The hypothesis is right in mechanism, but the lever is not reward scale.**
   - SAC's log-probability includes the tanh Jacobian. So the entropy term pushes the pre-tanh mean toward 0 with a gradient of 2α·a, while the critic's push is ∂Q/∂a·(1−a²), which vanishes at the bound (§2.1).
   - With our pedal map, holding pedal ≥ 0.95 needs a critic slope ∂Q/∂a ≥ 12.8 α. Pedal ≥ 0.99 needs ≥ 66 α. Pedal 1.0 needs an infinite slope.
   - With auto-α, scaling the whole reward is cancelled by α. A potential-based speed term does not change ∂Q/∂a at convergence at all (Ng 1999; Wiewiora 2003).
   - The levers are therefore: **(i) where the action map saturates**, **(ii) α / target entropy**, and **(iii) how sharply the critic resolves the throttle gain** (n-step returns).
2. **[R] The brake dabs have the same cause.** Inside the brake deadzone (a < −0.5) the critic slope is zero. Entropy then pulls the brake mean up to the deadzone edge at −0.5, where any noise or small actor error gives a brake touch. Every published GT-level agent avoids this: they use one combined throttle/brake axis or forbid using both pedals together.
3. **[S] The closest precedent:** Toyota Research Institute (TRI24) maps longitudinal +0.5 (not +1) to 100 % throttle. Their reason is that this "makes it possible for the agent to consistently use 100% throttle like humans do" (paraphrase verified, §1).
4. **[S] GT Sophy and Fuchs et al. used a fixed, small temperature.** GT Sophy used α = 0.01. Fuchs et al. used reward scale 1/α = 100. Both rewarded metres of progress at 10 Hz. Neither used auto-α with target −dim(A).

Recommendation, ranked (details in §5):
- (A) top-end throttle remap.
- (B) throttle/brake mutual exclusion.
- (C) late-stage lower target entropy or fixed small α.
- (D) a ∂Q/∂a diagnostic before and after.

Do not raise the learning rate, and do not add a throttle reward.

---

## 1. Racing RL: how throttle/brake, entropy and speed reward were set

| Work | Throttle/brake parameterisation | Rate / hold | Entropy / temperature | Speed reward | Throttle-at-bound evidence |
|---|---|---|---|---|---|
| **GT Sophy**, Wurman et al., *Nature* 602 (2022), doi:10.1038/s41586-021-04357-7. Read from the author PDF (Methods p. 223–224, Ext. Data Fig. 2). | [S] "the agent was presented control over the throttle and brake as one continuous action dimension". [S] "+1" = max throttle, −1 = max braking, 0 = neither. [S] Squashed normal: "The squashed normal distribution enforced sampled actions to always be within the [−1, 1] action bounds". [S] Covariance "constrained to lie in the range (e^−40, e^4)". | [S] 10 Hz. [S] "found no substantial performance gains from acting more frequently than 10 Hz". | [S] QR-SAC. [S] "the SAC entropy temperature value α was set to 0.01" (fixed). [S] lr 5.0×10⁻⁵ (Q) and 2.5×10⁻⁵ (policy); γ = 0.9896; batch 1,024; N-step backups. [S] "using N-step backups greatly improved performance compared with a standard one-step backup" (Fig. 2d). | [S] Course progress R_cp = s′_l − s_l (metres along the centreline). Off-course and wall penalties ∝ (Δt)·v_kph². | [R] Ext. Data Fig. 2c (Sarthe, corner 20) shows Sophy's throttle trace sitting at ~1.0 before and after the corner, in 10 Hz steps. This is a visual reading of the figure; no number is given. |
| **FUCHS21**: Fuchs, Song, Kaufmann, Scaramuzza, Dürr, "Super-Human Performance in Gran Turismo Sport Using Deep RL", *IEEE RA-L* 6 (2021), arXiv:2008.07971v2. | [S] One combined signal ω ∈ [−1, 1], "ωt=1 denotes full throttle and ωt=−1 full braking" (Sec. III-B). [S] "we apply a tanh activation to the output layer as proposed in" SAC (Sec. VI-E). [S] Combined because "the best human strategies do not involve the simultaneous use of throttle and brake" (VI-E). | [S] "limit the command frequency to 10 Hz during training" (VI-B). | [S] Table II: "reward scale (1/α) 100" (fixed temperature, SAC-v1 style). [S] lr 3×10⁻⁴, batch 4,096, γ = 0.98/0.982, and a "5-step TD error to stabilize training" (VI-C). | [S] Centreline progress cp_t − cp_{t−1}, plus wall penalty −c_w‖v‖² with c_w = 5×10⁻⁴. | Not discussed. |
| **LEE25**: Lee et al., "A Champion-level Vision-based RL Agent for Competitive Racing in GT7", *IEEE RA-L* 2025, arXiv:2504.09021. | [S] One axis "within the range [−1,1], with −1 indicating full braking and 1 indicating full throttle" (III-B). | [S] 10 Hz agent, 60 Hz game. | [S] "an entropy coefficient of 0.01". [S] lr "2.5×10−5" (IV-B). | [S] Progress weight λᵖ = 1.0. | Not discussed. |
| **TRI24**: Subosits et al. (TRI), "Towards an Autonomous Test Driver: High-Performance Driver Modeling via RL", arXiv:2412.03803 (Dec 2024). | [S] Two-dimensional action. [S] "A longitudinal input of -1 is mapped to the maximum brake pedal pressure observed in human demonstrations … while a value of 0.5 maps to 100% throttle" (III-A). [S, paraphrase] The asymmetry reflects that the throttle pedal is pushed against a physical stop, so the agent can use 100 % throttle consistently, like humans. [S] "the agent cannot use both the brake and throttle simultaneously". | [S] 10 Hz (simulator at 2000 Hz). | [S] DSAC (distributional SAC). Temperature not given in the text I read. | [S] "meters of progress along the track centerline per time step". [S] "The speed with which the throttle was released or the brake pedal applied was not penalized". | **The only source I found that remaps throttle specifically to make 100 % reachable.** |
| **ACGYM24**: Remonda et al., "A Simulation Benchmark for Autonomous Racing with Large-Scale Human Data", NeurIPS 2024 D&B, arXiv:2407.16680. Repo `dasGringuen/assetto_corsa_gym`, `main` branch, fetched 20 Nov 2026 (commit hash not visible on the page). | [S] Separate throttle, brake, steer, "all normalized to [−1,1]". [S] Delta (relative) actions that "greatly reduce oscillation". [S] `config.yml`: `use_relative_actions: True`, `max_steer_rate: 900.0`. | [S] 25 Hz. | [S] `algorithm/discor/.../sac.py`: auto α from log α = 0 (α₀ = 1); `self._target_entropy = -float(self._action_dim)` "# Target entropy is -\|A\|". [S] γ 0.992, n-step 3, lr 3e-4 (config.yml). | [S] r = v·(1 − d/12), proportional to forward speed. | [S] Table 2: SAC within 0.02–0.86 s of the best human. One example: F317 at Barcelona, 97.54 s vs 97.52 s. Throttle behaviour not discussed. [S] SAC fine-tuning across tracks was omitted because it "perform[ed] poorly in this setting". |
| **Jaritz et al.**, "End-to-End Race Driving with Deep RL", ICRA 2018, arXiv:1807.02371. | [S] Discrete: "broken into 32 classes" (Table I). Gas ∈ {0, 0.5, 1} × 9 steering values, plus hand-brake and brake classes. | — | A3C (discrete; no tanh issue). | [S] R = v(cos α − d). | [R] Full throttle is one of three discrete gas levels, so it is trivially reachable. |
| **Linesight** (Trackmania, IQN). Repo `Linesight-RL/linesight`, `config_files/inputs_list.py`, `main` fetched 20 Nov 2026. | [S] 12 discrete actions; accelerate is a boolean (e.g. index 0 "Forward" = accelerate only). | — | Value-based (no entropy bonus). | Progress-based (see `driving_agent_sota.md`). | [R] Binary accelerate; full throttle is the default action. |
| **Formula RL**: Remonda et al., IJCAI 2019 SURL workshop, arXiv:2104.11106. | [S] Continuous action space; DDPG variants. The abstract does not say whether throttle and brake are combined. | | | | Not opened beyond the abstract. |
| **AWS DeepRacer** (docs, `docs.aws.amazon.com/solutions/latest/deepracer-on-aws/action-space-reward-function.html`, via search snippet). | [S] The action is a *speed command* (min/max in mm/s) plus steering. SAC requires the continuous space; PPO allows discrete. | | | | [R] Not transferable: the action is speed, not pedal. |
| **"Learning to race in hours" (Chen et al.)** | **Could not identify.** Two searches found no such paper. The closest are Chisari et al., arXiv:2011.13332 (SAC, miniature car, ~50 h simulated driving) and Samak et al., arXiv:2110.05437 ("under 20 hours"). I did not open either beyond the search snippets. | | | | |

[R] Pattern:
- Every super-human GT agent (Sophy, FUCHS21, LEE25) and TRI24 uses **one longitudinal axis, or forbids simultaneous pedals**.
- Each uses a **fixed small α (0.01)** or reward scale 100, with **metres of progress per 100 ms** as the reward.
- Each uses **N-step targets** (5-step in FUCHS21, N-step in Sophy, 3-step in ACGYM24).
- None reports a partial-throttle problem.
- The only auto-α / −|A| racing agent (ACGYM24) used separate pedals with delta actions, and says nothing about throttle saturation.

---

## 2. SAC actions at the bounds

### 2.1 Mechanism [S + R]

- [S] **SAC log-likelihood with tanh**: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905, App. C, Eq. 26: log π(a|s) = log μ(u|s) − Σᵢ log(1 − tanh²(uᵢ)). The target entropy heuristic is "−dim(𝒜)" (App. D, Table 1).
- [S] **SB3 v2.9.0** implements this exactly.
  - `common/distributions.py`, `SquashedDiagGaussianDistribution`, `epsilon = 1e-6`: `log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)`.
  - `mode()` returns `th.tanh(self.gaussian_actions)`.
  - `sac/sac.py`: actor loss `(ent_coef * log_prob - min_qf_pi).mean()`; `ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()`; `target_entropy='auto'` → `-prod(action_space.shape)`.
  - **There is no action-bound penalty term in SB3 SAC.** The critic and actor losses contain none. The only boundary cost is the Jacobian inside `log_prob`.
  - `sac/policies.py`: `LOG_STD_MIN = -20`, `LOG_STD_MAX = 2`. A `Hardtanh(clip_mean=2.0)` on μ applies **only when `use_sde=True`**. That would cap the action at tanh(2) = 0.964, i.e. pedal 0.976. It does not apply to us.
- [R] **Derivation.** Reparameterise u = μ + σε, a = tanh(u). The Gaussian part of log π does not depend on μ. So:
  ∂(actor loss)/∂μ = 2α·tanh(u) − ∂Q/∂a·(1 − tanh²u).
  In the small-σ limit the throttle mean settles where **∂Q/∂a = 2α·a / (1 − a²)**. With our map a = 1.5·pedal − 0.5:

  | target pedal | a | u = atanh(a) | entropy cost −log(1−a²) [nats] | required ∂Q/∂a |
  |---|---|---|---|---|
  | 0.90 | 0.85 | 1.26 | 1.28 | 6.1 α |
  | 0.95 | 0.925 | 1.62 | 1.94 | 12.8 α |
  | 0.99 | 0.985 | 2.44 | 3.51 | 66 α |
  | 1.00 | 1.0 | ∞ | ∞ (capped at 13.8 by eps 1e-6) | ∞ |

  - With auto-α, H(π) is held at −3 nats overall (≈ −1 per dimension).
  - A saturated throttle spends 2–3.5 nats of that budget. The budget must be repaid by a wider pre-tanh σ (wide samples) or by more steering and brake noise. Both cost lap time.
  - So the constraint is structural, not a tuning accident.
- [S] **Executed-space entropy pulls the mean inward.**
  - He, Zhou, Wang, Xue, Fei, "Where Entropy Is Measured Matters: Policy Geometry in Bounded Continuous-Control PPO", arXiv:2608.24488 (25 Aug 2026): with entropy on the squashed action, "gradient descent therefore pulls the mean inward". Latent entropy "contributes no direct mean force" (Sec. 3.3).
  - Dog-Stand near-bound mean occupancy (P10): latent entropy 36.99 %, no entropy 22.25 %, executed entropy 1.91 % (Table 3).
  - PPO with a fixed coefficient of 10⁻³ and 3 seeds. Same mechanism as SAC's log-prob, different algorithm.
- [S] **The entropy term's main job in SAC is to stop saturation.**
  - Wang, Wu, Vuong, Ross, "Striving for Simplicity and Performance in Off-Policy DRL", ICML 2020 (PMLR 119), arXiv:1910.02208: "the entropy term in SAC addresses action saturation due to the bounded nature of the action spaces" (abstract).
  - Sec. 3.2: with SAC on Humanoid "the |μk| values are small, mostly in the range [-1.5,1.5]". Without entropy, "most of which are well outside the interval [-10,10]".
  - [R] tanh(1.5) = 0.905 → our pedal 0.937, which is below our 0.95 "full throttle" threshold. Humanoid is not our task, so treat this only as an order-of-magnitude illustration.
- [S] **Forcing saturation is not free.**
  - Shamass, "Unthrottling the Tanh Jacobian in SAC: A Negative Result on Bang-Bang Control and MetaDrive", arXiv:2609.09478 (8 Sep 2026, single author, preprint). Setting: SB3 SAC on a minimum-time double integrator (Δt 0.05, 200 steps, 10 seeds).
  - Vanilla SAC reached −31.6 vs the bang-bang reference −30.3, with p95|a| = 0.97.
  - Adding the critic gradient without the (1−a²) factor drove saturation to 0.99 but return collapsed to −195.5.
  - Quotes: "Saturating a bound is not the same as solving a problem whose optimum lives on that bound." "Removing entropy to 'let the bypass work' discards a stabilizer that the baseline needs."
  - [R] Do not hack the actor gradient. Move the bound instead (§5 A).

### 2.2 Bang-bang and alternative distributions

- [S] Seyde et al., "Is Bang-Bang Control All You Need? Solving Continuous Control with Bernoulli Policies", NeurIPS 2021, arXiv:2111.02552.
  - Gaussian MPO policies "tend to converge to bang-bang behavior" (Sec. 5.2, Fig. 3).
  - Bernoulli {−1, +1} policies "perform on par" on DMC.
  - Action penalties "help to mitigate bang-bang behavior" (Sec. 5.4.2), i.e. they pull away from bounds.
  - No SAC action histograms are given.
  - [R] DMC tasks are not entropy-budget-limited the way a 3-D racing policy with a −3 target is. The paper also shows that what decides bound-use is the reward structure ("maximum state reward" → bang-bang).
- [S] Seyde et al., "Solving Continuous Control via Q-learning", arXiv:2210.12566: bang-bang discretisation + value decomposition "matches performance of state-of-the-art continuous actor-critic methods". Venue not confirmed from the page.
- [S] Chou, Maturana, Scherer, "Improving Stochastic Policy Gradients in Continuous Control with Deep RL using the Beta Distribution", ICML 2017 (PMLR 70:834-843): clipping a Gaussian at bounds "introduces a bias". "the Beta policy is bias-free and provides significantly faster convergence" (TRPO/ACER).
  - [R] Not in SB3 SAC. A Beta with α, β > 1 has zero density at the bound, so it does not solve "exactly 1.0" either.
- [S] SB3 docs (`guide/rl_tips.html`): clipping is "a bandage"; "A better solution would be to use a squashing function (cf SAC) or a Beta distribution".
- [S] Kanervisto, Scheller, Hautamäki, "Action Space Shaping in Deep RL", IEEE CoG 2020, arXiv:2004.00980: "domain-specific removal of actions and discretization of continuous actions can be crucial for successful learning".

### 2.3 Target entropy choices

- [S] −dim(A) is the SAC default heuristic (Haarnoja 2018b, Table 1). SB3 and ACGYM24 use it. RLPD uses −|A|/2 (higher; see `driving_agent_sota.md`).
- [S] The temperature is sensitive. Haarnoja 2018b Sec. 1: "the scaling factor has to be compensated by the choice a of suitable temperature" and "a sub-optimal temperature can drastically degrade performance". Meta-SAC (Wang & Ni, ICML 2020 AutoML workshop, arXiv:2007.01932): "SAC is very sensitive to this hyperparameter". Target-entropy annealing exists for discrete SAC (Xu et al., NeurIPS 2021 DRL workshop, arXiv:2112.02852).
- [S] Raffin (SB3 maintainer) blog, 10 Feb 2025 and 1 Jul 2025 (araffin.github.io/post/sac-massive-sim, /post/tune-sac-isaac-sim): SAC only worked after "a smaller entropy coefficient" (`ent_coef="auto_0.006"`, later tuned `auto_0.0095`). He also narrowed action bounds to the range a trained PPO policy actually used (2.5–97.5 percentiles). Setting: IsaacLab quadruped, 1024–2048 envs.
- [R] I found **no** racing source that lowered the target entropy below −dim(A). The racing SOTA used a fixed α = 0.01 instead. In our terms, the corresponding knob is a lower target, e.g. −6 to −9 (−2 to −3 nats per dimension, roughly σ_action ≈ 0.03 to 0.012 when unsaturated), or freezing α at a small value late in training.

---

## 3. Continuing from a checkpoint with a higher LR or a changed reward

- [S] **Primacy bias**: Nikishin, Schwarzer, D'Oro, Bacon, Courville, ICML 2022, arXiv:2205.07802.
  - Definition: "a tendency to overfit initial experiences that damages the rest of the learning process".
  - For SAC, "we reset agent's networks entirely" every 2×10⁵ steps, and "The replay buffer is preserved between resets".
  - DMC IQM: 656 (SAC + resets) vs 501 (SAC), Table 1.
  - Where no primacy bias exists, resets cause spikes but "the agent is able to return to the previous state in just a few thousands of steps".
- [S] **LEE25 used a reset in a racing agent**: "To encourage the network to relearn overlooked features … we reinitialized the networks at 2,000 epochs, which aligned with the replay buffer reaching capacity" (of 20,000 epochs; Sec. IV-B / III-E).
- [S] **Plasticity.**
  - Dormant neurons and ReDo: Sokar et al., ICML 2023, arXiv:2302.12902.
  - LayerNorm + weight decay "highly effective at maintaining plasticity": Lyle et al., arXiv:2402.18762 (2024; the arXiv page lists no venue).
  - A "simple Soft Actor-Critic agent, appropriately regularized" beats prior SOTA: Nauman et al., ICML 2024, arXiv:2403.00514.
  - Warm-starting generalises worse than fresh initialisation unless settings that "damage the wall-clock gains": Ash & Adams, NeurIPS 2020, arXiv:1910.08475.
- [S] **Warm restarts**: SGDR, Loshchilov & Hutter, ICLR 2017, arXiv:1608.03983: "a simple warm restart technique for stochastic gradient descent" aimed at anytime performance, in supervised learning. I found no RL source that uses an LR warm restart to fix a converged-policy bias.
- [S] **Changing reward with an off-policy buffer**: PEBBLE (Lee, Smith, Abbeel, ICML 2021, arXiv:2106.05091): "we relabel all the agent's past experience when its reward model changes".
- [S] **Potential-based shaping**: Ng, Harada, Russell, ICML 1999 (pp. 278–287, mlanthology.org/icml/1999/ng1999icml-policy; I did not re-open the PDF today) proved policy invariance under F = γΦ(s′) − Φ(s). Wiewiora, JAIR 19 (2003), arXiv:1106.5267: shaping and Q-initialisation "make the same updates throughout learning".
- [R] **Consequences for us.**
  - Scaling the speed *potential* Φ_v changes the optimal Q only by a state offset −ΔΦ(s). So ∂Q/∂a is unchanged at convergence, and the throttle equilibrium of §2.1 is unchanged. It helps only transiently, through faster credit assignment.
  - With auto-α, scaling the *whole* reward by k is cancelled by α → kα (constrained formulation). The entropy level stays at −3.
  - **Scaling speed reward is therefore not the right lever for the last few % of throttle.**
  - If you do change the reward, relabel the buffer (or drop it). Then run a critic-only warm-up before unfreezing the actor (the repo already has a freeze/unfreeze path, `train_config.py:144`). A potential-only change shifts the critic by a state offset, so the actor gradient is undisturbed and a short warm-up should suffice.
  - **Raising the LR goes against the racing SOTA.** Sophy used 2.5–5×10⁻⁵ and LEE25 2.5×10⁻⁵. It also moves Adam steps proportionally at once. I found no evidence it fixes an entropy-held mean.
  - If plasticity is the suspicion, the sourced option is a reset with the buffer kept (Nikishin; LEE25), not a higher LR.

---

## 4. Remedies that fixed partial throttle in practice

| Remedy | Evidence | Status |
|---|---|---|
| Map full throttle below the tanh bound | TRI24: +0.5 ↦ 100 % throttle, "to consistently use 100% throttle like humans do" | **Direct precedent** [S] |
| One longitudinal axis / no simultaneous pedals | Sophy, FUCHS21, LEE25, TRI24 [S]. FUCHS21 gives the human-strategy reason [S] | Precedent. Removes the brake-deadzone edge [R] |
| Fixed small α | Sophy α = 0.01; FUCHS21 1/α = 100; LEE25 0.01 [S] | Precedent (but fixed from the start, not as a late fix) |
| N-step targets | Sophy ("greatly improved"), FUCHS21 (5-step), ACGYM24 (3) [S] | Precedent for credit assignment; we already support `n_steps` |
| Discrete / bang-bang throttle | Jaritz, Linesight (discrete); Seyde 2021/2022 [S] | Large change |
| Reward term on throttle | None found in racing. TRI24 explicitly does *not* penalise throttle release [S]. Seyde: action terms change bound-use [S] | Not recommended [R] |
| Deterministic eval | SB3 `mode()` = tanh(μ) [S]. [R] This is the *median* of the squashed distribution, not its mean or mode. Stochastic rollouts sit lower than tanh(μ) near +1 (concavity of tanh) | Make sure the 0–56 % figure comes from deterministic laps |
| Gradient bypass of the tanh Jacobian | Shamass 2026: saturation 0.99 but return −195.5 vs −31.6 [S] | Avoid |

---

## 5. Recommendation (interpretation, ranked by applicability)

These are options for Ibrahim; nothing here is decided.

**D. First, a cheap diagnostic (no training).** On deterministic Barcelona laps, log on each straight:
- the throttle and brake pre-tanh μ;
- α (`ent_coef`);
- g = ∂Q_min/∂a_throttle by autograd through the critic at a = tanh(μ).

Read the result as follows:
- If g ≈ 2αa/(1−a²) at the observed a, the entropy equilibrium of §2.1 explains it, and A and C will work.
- If g ≈ 0 or its sign flips along the straight, the critic does not resolve the gain. Then n-step and critic precision come first.
- A brake μ clustered just below atanh(−0.5) = −0.55 on straights confirms the deadzone-edge explanation of the dabs.

**A. Top-end throttle remap (env-side, this repo only; TRI24 precedent).**
- Example: keep `pedal = (a+0.5)/1.5` up to a knee (e.g. a = 0.7 → pedal 0.8), then go linearly to pedal 1.0 at a_sat = 0.85, and clip above.
- Required slope at a_sat drops from ∞ to ~6 α. Above a_sat, Q is flat, so the mean parks at a_sat and the deterministic action gives full throttle.
- Warm-start cost: actions above the knee in the buffer now mean a different pedal. Relabel them with a_new = map_new⁻¹(map_old(a_old)) or drop them.
- Partial-throttle resolution below the knee is unchanged.

**B. Brake/throttle mutual exclusion (Sophy / FUCHS21 / TRI24).**
- Ignore brake while throttle pedal > x, or ignore whichever pedal is smaller. Keeps the 3-output interface.
- A true combined axis changes the actor shape and the "outputs steering, throttle, brake" convention in `PROJECT_CONTEXT.md`. That is Ibrahim's call.

**C. Late-stage entropy.**
- Either lower `target_entropy` to about −6 to −9, or freeze α at its current value / 2–5× smaller for the speed phase (racing SOTA: fixed α = 0.01 with metre-scale progress at 10 Hz).
- Compare our α with our per-decision reward magnitude. Sophy's α / reward ratio is about 0.01 / 5–8 m.

**Not recommended:**
- higher LR;
- a throttle bonus or a brake-on-straight penalty (non-potential, biases the optimum; TRI24 avoids pedal-change penalties);
- re-scaling the speed potential (no asymptotic effect, §3);
- a Beta or Bernoulli policy for now (large change, no racing evidence).

---

## Sources (opened 20 Nov 2026 unless noted)

- Wurman et al. 2022, Nature: https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf (Methods; Ext. Data Fig. 2)
- Fuchs et al. 2021: https://arxiv.org/html/2008.07971v2
- Lee et al. 2025 (GT7): https://arxiv.org/html/2504.09021
- Subosits et al. 2024 (TRI): https://arxiv.org/html/2412.03803
- Remonda et al. 2024 (Assetto Corsa Gym): https://arxiv.org/html/2407.16680; https://github.com/dasGringuen/assetto_corsa_gym (`config.yml`, `algorithm/discor/discor/algorithm/sac.py`, main branch)
- Jaritz et al. 2018: https://ar5iv.labs.arxiv.org/html/1807.02371
- Remonda et al. 2019/2021 (Formula RL): https://arxiv.org/abs/2104.11106 (abstract only)
- Linesight: https://raw.githubusercontent.com/Linesight-RL/linesight/main/config_files/inputs_list.py
- DeepRacer action space: https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/action-space-reward-function.html (search snippet only)
- Haarnoja et al. 2018b: https://ar5iv.labs.arxiv.org/html/1812.05905
- SB3 v2.9.0: https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/stable_baselines3/sac/sac.py, `.../sac/policies.py`, `.../common/distributions.py`; https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- He et al. 2026: https://arxiv.org/html/2608.24488v1
- Wang et al. 2020 (ICML): https://arxiv.org/pdf/1910.02208
- Shamass 2026: https://arxiv.org/html/2609.09478
- Seyde et al. 2021: https://ar5iv.labs.arxiv.org/html/2111.02552; NeurIPS: https://papers.neurips.cc/paper_files/paper/2021/file/e46be61f0050f9cc3a98d5d2192cb0eb-Paper.pdf (listing only)
- Seyde et al. 2022: https://arxiv.org/abs/2210.12566
- Chou et al. 2017: https://proceedings.mlr.press/v70/chou17a.html
- Kanervisto et al. 2020: https://arxiv.org/abs/2004.00980
- Wang & Ni 2020: https://arxiv.org/abs/2007.01932; Xu et al. 2021: https://arxiv.org/abs/2112.02852
- Raffin 2025: https://araffin.github.io/post/sac-massive-sim ; https://araffin.github.io/post/tune-sac-isaac-sim/
- Nikishin et al. 2022: https://ar5iv.labs.arxiv.org/html/2205.07802
- Sokar et al. 2023: https://arxiv.org/abs/2302.12902; Lyle et al. 2024: https://arxiv.org/abs/2402.18762; Nauman et al. 2024: https://arxiv.org/abs/2403.00514; Ash & Adams 2020: https://arxiv.org/abs/1910.08475; Loshchilov & Hutter 2017: https://arxiv.org/abs/1608.03983
- Lee et al. 2021 (PEBBLE): https://arxiv.org/abs/2106.05091
- Wiewiora 2003: https://arxiv.org/abs/1106.5267; Ng et al. 1999: https://mlanthology.org/icml/1999/ng1999icml-policy/ (listing only)

Not opened / not found: "learning to race in hours" (Chen et al.) not identified; Chisari et al. arXiv:2011.13332 and Samak et al. arXiv:2110.05437 seen only as search snippets; GT Sophy Extended Data Table 1 (reward weights) and Supplementary Information not read.
