# Driving agent: state of practice for RL racing drivers (research note)

Prepared by `web-researcher`, 25 Oct 2026, for the SAC driving agent (OCD, F1-like car, 50 Hz, actions [steer, throttle, brake], soft ξref following, setup hidden, light fine-tune per setup).

**How to read this.** Plain statements in the tables and "Source says" bullets are what the cited paper or repo says, with section, table or file. Anything I derived (unit conversions, step counts, discount equivalents) is marked **(derived)**. Opinions and recommendations appear only under **Interpretation** headings. I note it wherever I could not open a source.

---

## 0. Sources actually opened

| Key | Source | How accessed |
|---|---|---|
| FUCHS21 | Fuchs, Song, Kaufmann, Scaramuzza, Dürr, "Super-Human Performance in Gran Turismo Sport Using Deep RL", IEEE RA-L 2021, arXiv:2008.07971 | arXiv HTML |
| SOPHY22 | Wurman et al., "Outracing champion Gran Turismo drivers with deep RL", Nature 602 (2022), doi:10.1038/s41586-021-04357-7 | Full text and Supplementary Information, from the author-hosted PDF at cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf. The image-only Extended Data Table 1 (reward weights) was **not readable**. |
| VASCO24 | Vasco, Seno, Kawamoto, Subramanian, Wurman, Stone, "A Super-human Vision-based RL Agent for Autonomous Racing in Gran Turismo", arXiv:2406.12563 (Jun 2024) | arXiv HTML (appendix with weights truncated) |
| LEE25 | Lee, Seno, Tai, Subramanian, Kawamoto, Stone, Wurman, "A Champion-level Vision-based RL Agent for Competitive Racing in GT7", IEEE RA-L 2025, doi:10.1109/LRA.2025.3560873, arXiv:2504.09021 | arXiv HTML |
| ACGYM24 | Remonda et al., "A Simulation Benchmark for Autonomous Racing with Large-Scale Human Data" (Assetto Corsa Gym), NeurIPS 2024 D&B, arXiv:2407.16680 | arXiv HTML + repo `dasGringuen/assetto_corsa_gym` @ `ef0fa502` (2 Aug 2025): `config.yml`, `assetto_corsa_gym/AssettoCorsaEnv/ac_env.py`, `sensors_ray_casting.py` |
| LINESIGHT | Linesight (Trackmania, IQN) | repo `Linesight-RL/linesight` @ `f6e7822f` (30 Sep 2026): `config_files/config.py`, `config_files/inputs_list.py`, `trackmania_rl/tmi_interaction/game_instance_manager.py`; docs site |
| TRI24 | Subosits, Lee, Manuel, Tylkin, Balachandran (TRI), "Towards an Autonomous Test Driver: High-Performance Driver Modeling via RL", arXiv:2412.03803 (Dec 2024) | arXiv HTML |
| PIA26 | Remonda, Xi, Hansen, Greiff, Lew, Talbot, Subosits, Wang, "Simulation Priors for Rapid Adaptation in Model-Based RL for Autonomous Racing", arXiv:2610.26757 (19 Oct 2026) | arXiv HTML |
| TCD23 | Ghignone, Baumann, Boss, Magno, "TC-Driver: Trajectory Conditioned Driving for Robust Autonomous Racing", Field Robotics 2023, arXiv:2205.09370 | arXiv HTML |
| HILD25 | Hildisch, Ghignone, Baumann, Hu, Carron, Magno, "Drive Fast, Learn Faster: On-Board RL for High Performance Autonomous Racing", arXiv:2505.07321 (RLC 2025 per RLJ listing) | arXiv HTML |
| TAL23 | Evans, Engelbrecht, Jordaan, "High-speed Autonomous Racing using Trajectory-aided Deep RL", arXiv:2306.07003 | arXiv HTML |
| TRAD26 | Leng et al., "TraD-RL: Expert Knowledge-driven RL for Autonomous Racing via Trajectory Guidance and Dynamics Constraints", arXiv:2603.05842v3 | arXiv HTML (first 100k chars) |
| BARI25 | Bári, Palkovics, "Self driving algorithm for an active four wheel drive racecar", arXiv:2506.06077 | arXiv HTML |
| AARK24 | Bockman, Howe, Orenstein, Dayoub, "AARK: An Open Toolkit for Autonomous Racing Research", arXiv:2410.00358v2 | arXiv HTML |
| APEX26 | Zhou et al., "Push the Limits: Human-Like Multimodal Extreme Embodied Intelligence for Autonomous Racing", Advanced Robotics Research (23 Oct 2026), doi:10.1002/adrr.202500210 | Wiley Scholar Gateway passages (Methods) only |
| CHIS21 | Chisari, Liniger, Rupenyan, Van Gool, Lygeros, "Learning from Simulation, Racing in Reality", ICRA 2021, arXiv:2011.13332 | abstract only |
| NG99 | Ng, Harada, Russell, "Policy invariance under reward transformations", ICML 1999, pp. 278–287 | MLAnthology abstract (the PDF text was not extractable) |
| GRZ17 | Grześ, "Reward Shaping in Episodic RL", AAMAS 2017 | secondary sources and the author's slides only (not the paper) |
| Algorithms | CrossQ (Bhatt et al., ICLR 2024, arXiv:1902.05605); DroQ (Hiraoka et al., ICLR 2022, arXiv:2110.02034); BRO (Nauman et al., NeurIPS 2024, arXiv:2405.16158); SimBa (Lee et al., ICLR 2025, arXiv:2410.09754); RLPD (Ball et al., ICML 2023, arXiv:2302.02948); WSRL (Zhou et al., ICLR 2025, arXiv:2412.07762); Primacy bias (Nikishin et al., ICML 2022, arXiv:2205.07802); gSDE (Raffin et al., CoRL 2021, arXiv:2005.05719); Bang-bang (Seyde et al., arXiv:2111.02552) | abstracts; RLPD and Nikishin full-text sections |
| Libraries | PyPI JSON (queried 25 Oct 2026); SB3 v2.9.0 source; SB3 changelog; sb3-contrib docs; SBX README; TorchRL repo tree; CleanRL repo; JAX install docs | direct |

---

## 1. Observation, action, reward, algorithm and budget of successful RL racing drivers

### 1.1 Comparison table

| | FUCHS21 (GT Sport) | SOPHY22 (GT Sport) | ACGYM24 (Assetto Corsa) | LINESIGHT (Trackmania) | TRI24 (Dymola GT3) |
|---|---|---|---|---|---|
| **Track preview** | 13 rangefinders every 15° over the front 180°, 100 m max; centreline curvature at N=10 samples spaced 0.2 s, from 1.0 to 2.8 s ahead (Sec. III-B) | 60 equally spaced 3D points on each of left edge, centreline and right edge; span scales with speed to cover ≈ next 6 s; egocentric frame (Approach section) | 12 rays over the front 180°, 200 m max (`sensors_ray_casting.py`: `N_RAYS=11` "plus one", `MAX_RAY_LEN=200`); reference-line curvature, 12 values over 300 m ahead (`ac_env.py`: `CURV_LOOK_AHEAD_DISTANCE=300`, `..._VECTOR_SIZE=12`); signed distance to reference line (`gap`) | 40 reference-line points in the car frame, taking every 20th virtual checkpoint at 0.5 m spacing, so 10 m spacing and 400 m horizon **(derived)** (`n_zone_centers_in_inputs=40`, `one_every_n_zone_centers_in_inputs=20`, `distance_between_checkpoints=0.5`); plus a 160×120 grayscale image | 15 left and 15 right track-edge points in the vehicle frame, "roughly a 5 second horizon"; lateral position e_norm (±1 at the edges); heading vs centreline (Sec. III-A) |
| **Vehicle state** | 3D linear velocity, 3D linear acceleration, heading vs centreline, wall-contact flag | 3D velocity, 3D angular velocity, 3D acceleration, load on each tyre, tyre slip angles, progress as sin/cos, surface inclination, orientation vs centreline, off-course and wall flags (Methods: Features) | speed, gap, force feedback, RPM, accelX/Y, gear, yaw rate, local vx/vy, 4 slip angles (`obs_enabled_channels`), each scaled by a fixed constant | velocity and angular velocity in the car frame, per-wheel sliding/contact/damper state, gear (game_instance_manager.py) | vx, vy, yaw rate; steer/throttle/brake; longitudinal, lateral, normal and yaw accelerations; **plus vehicle setup parameters** |
| **Previous actions / history** | previous steering δ_{t−1} only | most recent steering, throttle and brake as seen by the game; no stacking | last applied steer/throttle/brake, plus the full basic observation of the previous 3 steps (`PAST_ACTIONS_WINDOW=3`, `add_previous_obs_to_state: True`); state is 125-dim (paper Sec. 4.2) | previous 5 actions (`n_prev_actions_in_inputs=5`) | none stated |
| **Normalisation** | not stated | z-score from the assumed uniform range of each feature (Methods) | divide by fixed constants (`obs_channels_info`) | `state_normalization.py` | divide by expected max abs value |
| **Action space** | 2-D: steering in [−π/6, π/6]; **combined** throttle/brake ω ∈ [−1, 1]; auto gearbox | 2-D: steering in [−1, 1]; **combined** throttle(+)/brake(−) in [−1, 1]; squashed Gaussian, log-std range (e^−40, e^4) (Methods: Actions) | 3-D **separate** steer/throttle/brake in [−1, 1], applied as **deltas** to current controls (`use_relative_actions: True`); steering rate limit 900 deg/s (`max_steer_rate`), pedals 12 per s **(derived)** (`ac_env.py` `controls_rate_limit`) | 12 discrete combinations of {accelerate, brake} × {left, none, right} (`inputs_list.py`) | 2-D: steering; **combined** pedal where −1 = max observed brake pressure and +0.5 = 100 % throttle; no simultaneous application |
| **Control rate** | trained at 10 Hz (60 Hz sim), evaluated at 60 Hz | 10 Hz. "We tested action frequencies from 5 Hz to 60 Hz and found no substantial performance gains from acting more frequently than 10 Hz" (Methods) | 25 Hz (`ego_sampling_freq: 25`); physics 300 Hz | 20 Hz (`ms_per_action = 10 ms × 5`) | 10 Hz (sim 2000 Hz) |
| **Reward** | progress along centreline Δcp; on wall contact −c_w‖v‖², c_w = 5×10⁻⁴ (Eq. 1, Table II) | linear combination of course progress (masked when off course); off-course −Δt_off·v_kph² (Sarthe: ∝ v_kph, ×2 in chicanes); wall −Δt_wall·v_kph²; tyre slip −Σ min(\|slip ratio\|,1)·\|slip angle\|; plus racing terms (passing, collisions) | r = v·(1 − \|gap\|/12)/300 (`get_reward`, `gap_const: 12`); off-track **penalty disabled** (`enable_out_of_track_penalty: False`); optional action-diff penalty (off by default) | +5/500 per metre of centreline progress, −6/5000 per ms of time (`reward_per_m_advanced_along_centerline`, `constant_reward_per_ms`); a shaped term toward the current virtual checkpoint (`shaped_reward_dist_to_cur_vcp = −0.1`) | R = R_s (centreline metres, only on the racing surface) + R_edge = q₁·e_norm·v² when off track + R_slip = q₂·v·(excess front + rear slip) + linear penalties on steering rate and pedal changes beyond human maxima; q values not given (Sec. III-B, Eqs. 1–3) |
| **Termination** | none stated; 100 s rollouts; random start positions at 100 km/h | 150 s episodes (Methods: Training) | ≥ 3 wheels out of track; speed < 5 km/h for > 2 s (paper Sec. 4.2; `enable_out_of_track_termination`) | no checkpoint passed in 2 s (`cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2000`); 300 s cap | "too far off track or stops", with an extra penalty; episode ≈ 1.25× lap time; random start on centreline at half lap speed |
| **Algorithm and network** | SAC (Spinning Up base), policy + 2 Q + V nets, 2×256 ReLU, 599,566 params; 5-step TD | QR-SAC (32 quantiles, 7-step TD), **4×2048** ReLU, dropout 0.1 on policy (Suppl. Tables 1–2) | SAC (discor.pytorch base), 3×256 for policy and Q, 2 Q nets, 3-step TD | IQN (discrete), image CNN + float MLP (256), dense 1024 | DSAC (distributional SAC); sizes not given |
| **Hyperparameters** | lr 3e-4; batch **4096**; buffer 4×10⁶; 5120 updates per epoch; γ 0.98 / 0.982; reward scale 1/α = 100, so α = 0.01 **(derived)** (Table II) | lr Q 5e-5, policy 2.5e-5; γ 0.9896; α = 0.01 fixed; τ 0.005; batch 1024; buffer 10⁷; 6000 gradient steps per epoch; start after 40k transitions; critic grad-norm clip 10 | lr 3e-4 (policy, Q, entropy); batch 128; buffer 8×10⁶ (config) / 10⁷ (paper); γ 0.992; τ 0.005; update interval 1 (UTD 1); start 2000 steps; entropy target not reported | lr 1e-3 → 5e-5 → 1e-5 (schedule); batch 512; γ 0.999 → 1; n-step 3; soft τ 0.02 | not given |
| **Training budget** | 4 PS4 × 20 cars; 56–73 h per seed; ≈ 2151 epochs; 946,453 km driven; ≈ 1.7×10⁸ steps at 10 Hz **(derived)** | time trial: 10 PS4 (20 cars each) for ≈ 8 days; learned to lap in "a few hours", beat 95 % of humans "within a day or two", then 9+ more days and > 45,000 driving hours for the last tenths | ≈ 2M steps/day for 4 days; ≈ 8M steps ≈ 89 h of driving at 25 Hz **(derived)** | not stated | ≈ 2 days on an RTX 3080 Ti workstation, or 6 days on AWS g4dn.8xlarge; parallel simulators; steps not stated |
| **Result** | Audi TT, Brands Hatch: 1:15.913 vs best human 1:16.062 (Table I) | beat best humans in all three time trials and won head-to-head races | Dallara F317: SAC from demonstrations ≈ human best (e.g. Monza 104.08 s vs 104.31 s) (Table 2) | beat 10 of 12 targeted TMNF campaign world records (docs) | faster than a pro driver on all 4 setups at Fuji, e.g. baseline 96.63 s vs 98.38 s (Table I) |

### 1.2 Source says: other details worth keeping

- **SOPHY22 Fig. 2b ablation.** "Representing the upcoming track as sequences of points was advantageous". The compared variants were "Course points only", "Wall lidar, curvature" (the FUCHS21-style design) and "Projected position". Fig. 2a: without the QR enhancement to SAC, GT Sophy "would not be faster than the best human on Maggiore". Fig. 2d: a 7-step return beat 5-step. Fig. 2c: removing the off-course penalty gives slower laps and far fewer clean laps. There is also a "Fixed step cost" variant whose bar value I could not read from the extracted text.
- **FUCHS21 robustness (Sec. VI).** Zero-shot, the setting-A agent handled higher tyre friction with brief wall contact in two curves (+0.1 s). With lower friction it made contact in the three sharpest curves. On a different car it completed the lap but "sometimes overshoots" in sharp curves. On a different track it got stuck. After a game physics update (GTS 1.57, new tyres) it reached 1:14.686 after "5 days of training". The paper does not say whether this run was a fine-tune or a fresh start.
- **ACGYM24 Q4 (Table 3).** "SAC trained from scratch fails to complete a lap without the reference path." Sec. 3: n-step returns were "critical for learning". Sec. 4.2: "we do not apply absolute controls but rather deltas to the current controls, which we find to greatly reduce oscillation."
- **TRI24.** Four setups (baseline, understeer, oversteer, faster tyres) were trained in one policy, with setups drawn from "a circular queue" per episode, and the setup parameters are **in the state**. The agent ranked the setups in the same order as the pro driver. It was evaluated on 100 interpolated setups. Adding a path-imitation term R_imi = −(v(u) − μ(u))²/max(σ(u), σ_min)² changed lap time only slightly (96.54 s vs 96.69 s). Fine-tuning per setup is not done. Meta-learning for out-of-distribution setups is listed as future work.
- **VASCO24 / LEE25 (newer GT work).** 10 Hz control. Actions are **delta steering ±3°** plus a combined throttle/brake in [−1, 1]. Proprioception includes the last 3 steering angles and deltas. The critic is asymmetric: it sees the course points over 0.1–6 s, while the actor sees an image. LEE25 reward weights: progress 1.0, off-course 10.0, wall 20.0, steering-change 0.5, steering-history 5.0, plus racing terms. LEE25 QR-SAC: lr 2.5e-5, γ 0.9896, α 0.01, n-step 7, batch 512, buffer 5M, 4×2048 MLP. Networks are re-initialised once the buffer is full (citing Nikishin et al.).
- **HILD25 (on-board SAC, F1TENTH, 2025).** SB3-based SAC with an asynchronous actor/learner, 3-step TD, lr 3e-3, γ 0.96, batch 256, 2×256 network. The observation is 20 path points over 6 m plus the previous RL actions. Reward is λ·Δs (λ = 10) with penalty p = 10 on boundary violation. The agent learns within 10k–19k real steps (20–33 min). The authors report that 3-step TD reached lap completion before 10 min versus about 20 min for 1-step.
- **TCD23 (SB3 SAC, F1TENTH sim).** The observation is 30 trajectory points relative to the car, plus deviation, heading, vx, vy and yaw rate. The reward is progress − |n| (deviation), with −0.01 on violation. **Friction μ is randomized at each reset**, μ ~ N(1.0489, 0.0375). Under model mismatch the crash rate was 14.3 % vs 95.2 % for end-to-end SAC and 38.1 % for MPC (Table I). γ 0.99, batch 64, 1M steps.
- **AARK24 (Assetto Corsa).** SAC (Remonda base) with 300 m reference-line curvature preview, tyre slip, and distances to the line and to the limits. A speed-reward **curriculum** is used: "encourages agents to learn the racing line and complete laps before attempting to travel faster". SAC "could not complete five consecutive laps".
- **BARI25 (TORCS, SB3 PPO).** 20 Hz control; progress reward; −10 terminal penalty for leaving the track; laps first completed after about 15M steps; consistent laps after about 230M. This is PPO, much less sample-efficient than the SAC results above.
- **TRAD26 (2026).** Uses a trajectory-tracking reward 1/(1+l²), speed and heading terms, and a **two-stage curriculum** that removes the racing-line speed reference after 200k of 250k steps "to overcome the conservatism inherent in the prior trajectory". Lr 1e-4, γ 0.99, buffer 200k, MLP 3×256 after a CNN.
- **APEX26 (Yas Marina, real super-formula car).** Uses an SAC-like constrained policy iteration. The reward "penalizes the previewed deviation from the reference trajectory" (distance and heading errors), and the optimisation enforces an action-cost upper bound and an entropy lower bound. I saw only Methods excerpts; hyperparameters were not visible.

---

## 2. Action parameterization

### Source says
- **Combined longitudinal axis**: FUCHS21, SOPHY22, VASCO24/LEE25 and TRI24 all use one throttle(+)/brake(−) axis. SOPHY22 Methods explains why: "Because the throttle and brake are rarely engaged at the same time, the agent was presented control over the throttle and brake as one continuous action dimension." TRI24 maps the axis asymmetrically (−1 = max brake, +0.5 = full throttle) and forbids simultaneous use.
- **Separate pedals**: ACGYM24 and AARK24 use three separate outputs. In ACGYM24 they are delta (rate) actions.
- **Steering rate**: ACGYM24 uses delta actions with a 900 deg/s steering-wheel rate limit, and credits deltas with greatly reducing oscillation. VASCO24/LEE25 use delta steering ±3° per 0.1 s step, plus reward penalties on steering change and on steering sign-flip history. FUCHS21 imposed 0.03 rad per frame (60 Hz) and 80 % per frame pedal limits for the human-fairness evaluation. TRI24 penalizes steering rate and pedal changes beyond human maxima.
- **No controlled ablation** of combined vs separate pedals for racing turned up in my searches (two web searches plus papers above). That is an absence of evidence.
- **Exploration and smoothness, generic.** Seyde et al. (arXiv:2111.02552): trained continuous-control agents often converge to actions at the bounds (bang-bang), and Bernoulli bang-bang policies match state of the art on benchmarks. gSDE (Raffin et al., CoRL 2021, in SB3) replaces per-step noise with state-dependent noise, resampled periodically, to avoid "jerky motion patterns". It was tested on, among other robots, an RC car.

### Interpretation
- A combined axis removes one action dimension and the throttle+brake-at-once region, which carries no value for a racing car without left-foot-braking tricks. With SAC's per-dimension Gaussian noise, separate pedals spend roughly half of early exploration with both pedals partly pressed. Every top GT result used the combined axis. The project fixes the output as [steering, throttle, brake], and a combined policy output can be mapped to those three simulator inputs in the env (throttle = max(ω,0), brake = max(−ω,0)). The agent still "outputs" all three in the simulator sense. Whether that satisfies the spec is Ibrahim's call.
- The ACGYM24 and LEE25 evidence favours **steering as a rate/delta** (or at least a hard rate limit) at control rates above 10 Hz. At 50 Hz, absolute steering with SAC noise produces 50 Hz chatter, which in turn feeds the u̇ᵀWu̇ term of reward (1). A delta parameterization makes u̇ the action itself, so W penalizes the action directly. One trade-off: delta actions put the current control into the Markov state, so the current steer/throttle/brake must be in the observation, as ACGYM24 does.

---

## 3. Reward shaping for following ξref without biasing lap time

### Source says
- **NG99**: adding F(s,s′) = γΦ(s′) − Φ(s) leaves the optimal policy unchanged for any potential Φ. The abstract: "Furthermore, this is shown to be a necessary condition for invariance". It also links known shaping "bugs" to rewards that are not potential-based.
- **GRZ17 (via secondary sources and the author's slides)**: in episodic tasks the guarantee needs Φ(terminal) = 0. Otherwise the terminal potential changes the optimum. Early termination such as going off-track is exactly this case.
- **Racing practice: nobody in the opened papers uses a formally potential-based ξref term.**
  - FUCHS21, SOPHY22, TRI24 and LINESIGHT use centreline progress (Δs).
  - ACGYM24 multiplies speed by (1 − |gap|/12), a non-potential-based line term. Without it, SAC from scratch could not complete a lap.
  - TCD23 subtracts |n| (deviation).
  - TAL23 rewards matching a pure-pursuit tracker on the optimal line. At 6 m/s the baseline completed < 25 % of laps on three maps vs > 75 % for TAL, but TAL had slightly slower lap times.
  - TRI24's path-imitation term cost ≈ 0.1–0.2 s at Fuji (96.54 s vs 96.69 s for pure RL across variants, Sec. V).
  - TRAD26 and AARK24 use **curricula**: follow the line or the speed reference first, then remove or relax it.
  - APEX26 penalizes previewed deviation.
- **HPRS** (Berducci et al., arXiv:2110.02792) is a potential-based shaping framework with F1TENTH experiments. Its potentials come from safety, target and comfort specifications, not a racing line. I saw only the search snippet.
- **Off-track handling in the opened papers:**

| Paper | Off-track treatment |
|---|---|
| FUCHS21 | wall: −5e-4·‖v‖² per step in contact; no termination |
| SOPHY22 | off-course: −(Δt off-course)·v_kph²; progress masked while off course; wall: same form; game flag "3+ tyres out"; no termination |
| LEE25 | off-course weight 10, wall 20 (progress weight 1) |
| TRI24 | R_edge = q₁·e_norm·v²; terminate with extra penalty if far off |
| ACGYM24 | terminate at ≥ 3 wheels out; **no penalty** (config) |
| HILD25 | p = 10 with progress multiplier 10 per metre |
| BARI25 | −10 terminal |
| TCD23 | −0.01 |

### Interpretation (important for reward (1))
1. **A potential-based ξref term cannot make the driver follow ξref at convergence.** If the 14 Oct alternative uses Φ(s) = −k·e(s)² (or any Φ of the distance to ξref), then by NG99 the optimal policy is the same as with no ξref term at all, so the "driver axis" disappears at the optimum. That is the same outcome as c → 0. It only changes learning speed. So PBRS can serve as a **learning aid** (with Φ(terminal) = 0), while the c·e² term is what deliberately defines the driver. The choice between them is a formulation decision for Ibrahim, not a technicality.
2. **Progress is almost potential-based already.** a·Δs = a(s′ − s) is PBRS with Φ = a·s when γ = 1, and over a full lap it sums to a·L, a constant. Lap time therefore enters (1) mainly through −b·Δt (as in LINESIGHT's −6/5000 per ms). With γ < 1, a·Δs additionally rewards getting progress sooner, which also favours speed.
3. **Living-cost + termination trap.** At a fixed 50 Hz, b·Δt·(1 + u̇ᵀWu̇) is a per-step cost of at least b·Δt. When the per-step reward is negative (slow driving, large e, jerky inputs), ending the episode stops the losses. Off-track termination then becomes attractive unless the off-track penalty P exceeds the discounted cost the agent avoids. A safe rule is P ≥ b·Δt·(1 + max u̇ᵀWu̇)/(1 − γ) plus a margin, or make the terminal penalty ∝ v² like SOPHY22 and TRI24. This argument is mine; none of the papers states it.
4. **Off-track penalty scale**: the successful GT agents scale off-course and wall penalties with speed (v or v²), so cutting a corner at high speed always costs more than the progress it gains. The penalty also cannot be avoided by slowing to a crawl because of the time cost.

---

## 4. Fine-tuning / transfer to changed vehicle dynamics

### Source says
- **TRI24** avoids per-setup fine-tuning by putting the setup parameters in the state and training on all setups jointly ("less computationally expensive than individually training a policy for each"). Our driver may not see the setup, so this route is closed except through implicit inference (below).
- **FUCHS21** zero-shot: small friction changes cost about 0.1 s and caused some contacts; a car change produced overshoots. Retraining after the physics update took 5 days; the paper does not say whether it started from the old policy.
- **ACGYM24 (Sec. 5.2, Q2)**: SAC fine-tuning across tracks was omitted "due to poor performance". TD-MPC2 pretrained on other tracks completed laps "after a few episodes" vs hundreds from scratch. A track ID was added to observations.
- **PIA26 (TRI/UCSD, Oct 2026)**: model-based (TD-MPC2) offline-to-online adaptation.
  - The replay mixes offline (prior) and online data, with the offline weight annealed from 1 to 0.5.
  - Grip changes on Silverstone: SAC from scratch fails, while PIA reaches 112–118 s (Table I).
  - On the real car, under 1 h of data reached 100 % success and beat the MPC reference (49.55 s vs 52.47 s, Table III).
  - Their ablation randomized tyre grip, mass, engine torque and yaw inertia in the low-fidelity generator.
  - The paper does not test fine-tuning of a pretrained model-free policy.
- **TCD23**: randomizing μ during training cut crashes under model mismatch from 95 % (end-to-end) to 14 % (trajectory-conditioned). **CHIS21**: "model randomization" in simulation plus 3 h of real data matched MPC lap times with 50 % fewer violations (abstract only).
- **HILD25**: zero-shot transfer to a new track was unstable; one extra battery (≈ 20–30 min, ≈ 10–19k steps) of on-board training gave stable laps (7.93 s ± 0.13).
- **Generic offline-to-online results relevant to "keep the buffer?":**
  - **RLPD** (Sec. 4.1–4.3, App. B): symmetric sampling (50 % prior data / 50 % new data per batch), **LayerNorm in the critic** to bound value extrapolation, ensembles and UTD 20; batch 128+128, lr 3e-4, target entropy −|A|/2.
  - **WSRL** (ICLR 2025): at the start of fine-tuning, value functions can suddenly diverge when the data distribution shifts, which causes "unlearning". A short **warm-up of rollouts from the pretrained policy** recalibrates Q, and old data then need not be kept.
  - **Primacy bias** (Sec. 5.1): for SAC they reset the **entire** network every 2×10⁵ steps while "the replay buffer is preserved between resets". That remedies over-fitting to early data and is the opposite of what a light fine-tune wants.
  - **LEE25** reinitialises its networks once when the buffer is full.
- I found **no paper that reports step counts for lightly fine-tuning a pretrained SAC racing policy to a changed car setup**. Two searches plus the papers above turned up nothing. PIA26 and HILD25 are the closest, and both are tens of minutes of real driving, not setups.

### Interpretation
- The cheapest way to make "a few minutes" of fine-tuning enough is to make the base driver already competent on many setups, without telling it the setup:
  1. Train the base policy on **setups randomized over the search space**: domain randomization, as in TCD23, CHIS21 and the PIA26 ablation.
  2. Give it a **short history** (ACGYM24's 3-step stack, or the last few actions and accelerations) so it can infer the handling implicitly.
  3. The fine-tune then only polishes.

  This conflicts with DECISIONS.md #4 ("trains on p0 = Setup() defaults"), so it is a question for Ibrahim. Two caveats: randomization may make the base policy more conservative, and whether the setup should stay unseen when it can be inferred from history is a design question.
- For the fine-tune itself, the evidence points to:
  - keep actor and critic weights (no resets);
  - **keep the old buffer but sample it at most 50/50 with fresh data**, as RLPD and PIA26 do, or drop it after a WSRL-style warm-up;
  - **warm up** with a few thousand steps of the frozen policy before any gradient step;
  - use LayerNorm in the critic;
  - use a lower or the same lr, and freeze α or start from the learned α. GT agents used a fixed α = 0.01 with reward scaling.

  These are transfers from generic offline-to-online work, not racing evidence.
- **Fairness of setup scoring**: a fixed, small fine-tune budget favours setups close to the one the driver was trained on. To keep the comparison fair:
  - fix the budget in **environment steps**, not wall time;
  - evaluate the deterministic mean policy over several laps (as planned);
  - use ≥ 2 fine-tune seeds for the top setups;
  - log the lap-time curve during fine-tuning to check that it has flattened. The planned "re-rank top setups with longer fine-tunes" is the right safeguard.

---

## 5. Library choice on Apple Silicon M4 (Oct 2026)

### Source says (versions from PyPI JSON, queried 25 Oct 2026)

| Package | Latest | Released | Notes |
|---|---|---|---|
| stable-baselines3 | 2.9.0 | 2026-06-15 | Python ≥ 3.10. SAC defaults (v2.9.0 `sac/sac.py`): lr 3e-4, buffer 1e6, batch 256, τ 0.005, γ 0.99, `ent_coef="auto"`, `target_entropy="auto"`, `use_sde` (gSDE), and **`n_steps`** (NStepReplayBuffer, added in 2.7.0, 2025-07-25). `save_replay_buffer`/`load_replay_buffer` exist (`off_policy_algorithm.py`). `torch.compile` save/load fixed in 2.8.0, example added in 2.9.0. |
| sb3-contrib | 2.9.0 | 2026-06-15 | Algorithms: ARS, **CrossQ** (since 2.4.0, 2024-11-18), MaskablePPO, RecurrentPPO, QR-DQN, **TQC**, TRPO. CrossQ default lr 1e-3, batch 256. |
| sbx-rl (SBX, JAX) | 0.28.0 | 2026-07-24 | SAC, SAC-N, TQC, **DroQ**, **CrossQ**, TD3, DDPG, PPO, DQN; **SimBa** architecture (added in SB3 2.5.0 timeline); `layer_norm=True`; parameter resets; n-step |
| torchrl | 0.14.1 | 2026-10-10 | objectives include `sac`, `redq`, `crossq`, `tqc`, `td3`, `tdmpc2`; sota-implementations include sac, crossq, redq, tqc, offline_to_online |
| cleanrl | 1.2.0 (PyPI) | 2023-05-22 | single-file scripts; repo last commit 2026-04-20; `sac_continuous_action.py` defaults: buffer 1e6, γ 0.99, τ 0.005, batch 256, policy lr 3e-4, q lr 1e-3, policy_frequency 2, autotune α |
| torch | 2.14.1 | 2026-09-30 | |
| gymnasium | 1.4.0 | 2026-10-05 | |

- **Device on Mac.** SB3 v2.9.0 `utils.get_device` docstring: "For now, it supports only cpu and cuda." `device="auto"` → cuda → falls back to **CPU** on a Mac. An explicit `"mps"` string is passed to torch but is not a documented SB3 path. SB3 maintainer (issue #1245, Feb 2023, about A2C/PPO): "best is to use CPU (and tune the number of threads) as long as you are not using CNN." General MPS reports (PyTorch forums, Apple dev forum) show small networks running slower on MPS than on CPU because of kernel-launch overhead. These are anecdotal; I found no SAC-specific benchmark.
- **JAX (SBX) on Mac.** The JAX install page lists CPU on macOS aarch64 as supported and Apple GPU as "experimental", while also saying "JAX is not supported on Mac/OSX GPU". In practice SBX runs on CPU there.
- **SubprocVecEnv** (v2.9.0) uses `forkserver` if available, else `spawn`. On macOS the training script therefore needs the `if __name__ == "__main__":` guard, and the env (pybind11 OCD) must be constructible inside each worker.
- **SB3 SAC network options.** The actor and critic are built with `create_mlp(features_dim, -1, net_arch, activation_fn)` (`sac/policies.py`). LayerNorm or dropout (DroQ, RLPD, SimBa style) is not exposed through `policy_kwargs`, even though `create_mlp` itself accepts `post_linear_modules`. It needs a small custom policy subclass, or SBX/TorchRL.
- **What the newer algorithms claim** (abstracts):
  - CrossQ keeps **UTD = 1**, removes target networks, uses batch norm, and "substantially reduces the computational cost compared to REDQ and DroQ" (which use UTD 20).
  - DroQ matches REDQ's sample efficiency at roughly SAC compute.
  - BRO scales a regularized critic with optimism and reaches SOTA on 40 DMC/MetaWorld/MyoSuite tasks.
  - SimBa (observation normalization + residual blocks + LayerNorm) lets plain SAC match SOTA on DMC/MyoSuite/HumanoidBench.
  - All of these are benchmarked on locomotion/manipulation suites, not racing. None of the racing papers above uses them. The racing results come from plain SAC/QR-SAC/DSAC with **n-step returns** (3–7), large batches (1024–4096 for GT) and long training.

### Interpretation
- **Recommended: SB3 2.9.0 SAC on CPU**, with `n_steps=3–5` and `SubprocVecEnv`, as the baseline.
  - Mature API, replay-buffer save/load (needed for fine-tunes), gSDE, and the same API as sb3-contrib CrossQ. That makes **CrossQ an A/B experiment that costs one line**.
  - TCD23 and HILD25 are racing precedents built on SB3 SAC.
  - DroQ, REDQ and BRO raise **sample** efficiency by spending more gradient steps per env step. Our simulator is cheap (OCD ≈ 12–16 µs per integration step per PROJECT_CONTEXT), so wall-clock will be dominated by gradient updates on the M4 CPU. A "few-minute" fine-tune is a wall-clock budget, so high-UTD methods are likely the wrong trade. **This needs a sim-runner benchmark** (env steps/s vs gradient steps/s on the M4) before deciding.
  - CrossQ (UTD 1, no target nets) and a LayerNorm critic (cheap; RLPD's argument for prior-data fine-tuning) are the two upgrades that fit the budget.
- TorchRL (most features, most churn) and SBX (fast JAX, but a second framework) are reasonable alternatives but not needed for v1. CleanRL is good reading material, but its PyPI package is stale (2023, Python < 3.11).
- **γ at 50 Hz.** The papers' effective horizons are about 5–10 s: FUCHS21 0.98 @10 Hz ≈ 5 s; SOPHY22 0.9896 @10 Hz ≈ 9.6 s; ACGYM24 0.992 @25 Hz ≈ 5 s. The same horizons at 50 Hz need γ ≈ 0.996–0.998 **(derived)**: γ₅₀ = γ_f^(f/50). The alternative is an action-repeat (decide at 10–25 Hz, simulate at 50 Hz), which SOPHY22's "no gain above 10 Hz" finding supports.

---

## 6. Implications for us (interpretation, not source claims)

1. **Observation.**
   - **ξref and track-edge points** in the car frame, with spacing that scales with speed over about 5–6 s. At F1 speeds that is up to about 500 m, e.g. 15–30 points per line (SOPHY22, TRI24, LINESIGHT).
   - Car-frame vx, vy, yaw rate, ax, ay; the current applied steer/throttle/brake; signed e and heading error to ξref.
   - Optionally slip angles (SOPHY22, ACGYM24); they reveal setup effects without revealing the setup.
   - Add a short history (2–3 steps or the last few actions), as ACGYM24 and LINESIGHT do. No setup parameters.
2. **Action.** The policy outputs [steering rate or delta, combined throttle/brake]. The env maps these to the three OCD inputs, with a hard steering-rate limit. Confirm with Ibrahim that this still counts as "outputs steering, throttle, brake".
3. **Reward.**
   - Keep c·e² as the explicit driver term; a potential-based ξref term would remove the driver at the optimum.
   - Make the off-track penalty larger than the avoidable living cost, ideally scaled with v or v².
   - Mask progress while off track (SOPHY22).
   - Consider a short curriculum: larger c early, then the fixed c (AARK24, TRAD26). The final c stays constant per PROJECT_CONTEXT.
4. **Algorithm.**
   - SB3 2.9.0 SAC, CPU, n-step 3–5, 2–3×256 MLP, batch 256–1024, γ ≈ 0.996–0.998 at 50 Hz, or action-repeat to 10–25 Hz.
   - Run CrossQ (sb3-contrib) as the first A/B.
   - LayerNorm critic via a small custom policy.
5. **Fine-tune.**
   - Base-train with setup randomization (needs Ibrahim's OK; it conflicts with DECISIONS #4).
   - Fine-tune with weights kept, a warm-up rollout, a mixed old/new buffer (≤ 50 % old), and a fixed env-step budget.
   - Validate fairness by checking that the fine-tune curve has flattened, and re-rank top setups with longer fine-tunes.
6. **Gaps.** No source gives combined-vs-separate pedal ablations, nor step counts for light SAC fine-tunes after setup changes. Both are worth a small experiment of our own, and could be a thesis contribution.
