# Agent Context: RL Setup-Optimization Thesis (Ibrahim Khaled, AUB)

> Load this before doing anything on this project. It is the distilled state of every design discussion up to **25 Oct 2026**. Where something changed over time, the **current** decision is stated first and the superseded version is listed in §12 so you don't revive it. Anything marked **OPEN** is undecided: ask Ibrahim, don't pick for him.
>
> Architecture figure + car/driver/track/setup tables: `~/Downloads/Fianlizing_architecture.pdf` ("What a Car, a Driver, a Track and a Setup Are", Tables I–IX). Driving-agent decisions made in this repo are logged in `docs/DECISIONS.md`.

---

## 1. People and roles

| Who | Role |
|---|---|
| **Ibrahim Khaled** | Undergrad researcher, Computer Science in Engineering, AUB (Beirut), third year. Owns **vehicle setup optimization** (the setup agent, the simulator work, the car/track/setup definitions). |
| **Dr. Naseem Daher** (Prof. Naseem) | Supervisor. Decides simulator choice, baselines, validation approach. Ibrahim emails him for decisions. |
| **Jad Al Lakkis** | Friend and thesis partner. Owns the **driver side**: driver coaching/optimization and the driver's **reference lap ξref**. His paper will later be **merged** with Ibrahim's. |
| Mr. Sami El Ghossaini | Lab contact for the CarSim license and lab PC specs. |

---

## 2. The research problem

**One-line version:** Find the F1 car setup that gives a specific driver (defined by his reference line) the fastest lap on a given track, **model-free** with RL, and make the setup agent **generalize** to unseen cars, drivers and tracks.

**Long-run motivation:** how does vehicle setup interact with driver style to shape the optimal racing line and lap time, and how can a new car's setup be derived from a driver's historical best setup and style? No published work closes the loop *driver style → optimal setup → predicted line and lap time*.

**Confirmed gap (Sep 2026):** no paper combines (a) RL as the optimizer, (b) trackside race-car setup parameters as the decision variables the agent *outputs*, (c) lap time as the objective. Stacked on top: conditioning on a specific driver, and generalizing across unseen car/driver/track.

**vs Leyton-Romero & Limebeer (VSD 2026):** they use optimal control on an explicit vehicle model, for a **free-line time-optimal driver** on one track. Ibrahim's driver is **tied to a specific reference line**, the method is **model-free**, and **generalization** is the key addition. Keep the comparison accurate; don't overstate it.

**Why RL and not NLP/OCP:** NLP wins when the parameter→lap-time map is smooth and differentiable end-to-end; it breaks when the map runs through a non-differentiable or commercial simulator. Don't claim RL is generally superior. OCP is a **baseline only**.

---

## 3. Scope decisions (current)

- **Car class: Formula 1.** Target: **2017 Ferrari SF70H** in Assetto Corsa.
- **One driver per reference line.** The driving agent is a training/evaluation scaffold, not the product.
- **Only the setup is varied.** Each candidate setup gets a **light fine-tune** of the driving agent before scoring.
- **Fixed, out of scope:** engine/ERS at max, fuel, powertrain hardware, engine maps, DRS (closed), engine-brake rotary, tyre compound, tyre pressures, tyre temperature/wear.
- **Driving agent outputs only:** steering, throttle, brake.
- Brake bias stays a setup knob. Differential optional later. Left = right for every knob.

---

## 4. Architecture (current: "ProblemFinalize" + case definition v6)

```
Input (summaries) ──► Setup agent (TD3, deterministic) ──setup p──► Simulator (OCD)
  z_car (9)                 ▲  ⇅ knobs ↔ clicks (search space formulas)     │ obs ▲ actions
  z_drv (6)                 │                                               ▼     │
  z_trk (10)           Critic ("race engineer")                     Driving agent (SAC)
  p0   (26)            predicts J(p)  ◄── train on (p, J) ◄── Setup loss J ◄── B laps
                                                                    reward (1), follows ξref,
                                                                    does NOT see the setup
```

- **Setup agent:** deterministic actor (TD3, effectively γ = 0, bandit-like). One setup p per **run** (B laps). Input = compact summaries + p0 = 51 inputs.
- **Critic:** regresses J: `(Q(p) − J(p))²`. Only the best few proposed setups get driven.
- **Driving agent:** SAC. Sees upcoming track checkpoints, speed, position, last few observations, and ξref. Does **not** see the setup.
- **Driving reward (1), per step k:**
  `r_k = a·Δs_k − b·Δt_k·(1 + u̇_kᵀ W u̇_k) − c·e_k²`
  progress − time + control smoothness (sums to Limebeer's lap cost) − distance from ξref. Leaving the track ends the lap with a penalty. **c is constant and the same for all drivers.** Large c = copies ξref; small c = adapts the line to the setup; c → 0 kills the driver axis.
  - Alternative (14 Oct): potential-based "progress toward the line" (Ng et al. 1999).
- **Setup loss (2):** `J(p) = L̄(p) + λ·F(p)` — L̄ = lap cost averaged over B laps; F = variation of the four tyre loads and front/rear ride heights; λ small.
- **Training schedule:** driver converges on frozen p0 → critic trained on spread-out setups → setup-agent rounds alternating with light driver refreshes; stop when best lap improves < ~0.05 s; re-rank top setups with longer fine-tunes.
- **Production:** setup agent outputs a setup in one forward pass; fine-tune the driver on it for the trajectory; report deterministic policy mean over several laps. A new track first needs a p0 driver-training phase.
- Model size ≈ 1–1.4 M params total. M4 Mac for dev; L5 GPU for heavy runs.

### Input summaries (case definition v6)
- **z_car (9):** peak power, rpm of peak power, first gear ratio, top gear ratio, max brake torque, total C_zA, downforce front share, C_xA, ride-height sensitivity.
- **z_drv (6), averaged over corners of ξref:** apex position, entry offset, exit offset, brake point, min speed as % of grip-limited speed, trail-braking share. **The driver IS ξref.**
- **z_trk (10), from the track only:** slow/medium/fast corner shares, longest straight, straight share, braking zones, lap length, grip (`lambda_mue`), elevation range, roughness. Not yet computed.

---

## 5. Setup search space

One physical value + a ratio per group → Milliken Ch. 16 formulas → simulator settings. Code: `ocd_lap/car_setup.py`, `ocd_lap/translator.py` in ocd_first_lap.

**OPEN — 26 vs 16 knobs.** v6 (19 Oct) = 26 knobs (Table IX of the PDF). 25 Oct filter → 16 knobs: f_F, f_R, K_φ, χ, ζ_F, ζ_R, ρ, camber F/R, toe F/R, front height, rake, downforce level, downforce front share, brake bias. Confirm before changing code/docs. (Irrelevant to the driving agent: it never sees the setup.)

Formulas: `K_w = m_c(2πf)²`; `C = 2ζ√(K_w m_c)`; heave `K_h = 2m_c(2πf_h)²`; ARB `K_b = χK_φ − ½K_w t²` (≥ 0), `c_ar = 2K_b/t²`; m_c = 171 kg front, 199 kg rear.

---

## 6. Simulators

**Current:** **Open Car Dynamics (OCD, TUM)** for training; **Assetto Corsa (SF70H)** for final confirmation.

- **OCD:** C++20, Apache-2.0, native on M4 via custom pybind11 binding. RWD_LSD + PT1 steering + double-track + **MF 5.2** + ride-height aero. ~12–16 µs/integration step. Validated vs AV-21 (Sagmeister et al., IEEE IV 2024).
- Not used: Speed Dreams (GPL — never copy its code into Apache OCD), CARLA, F1TENTH, BeamNG/Gazebo/Chrono, CarSim (too slow/licensed).
- **AC:** real-time only; RL needs PC AC (assetto_corsa_gym). Ibrahim has PS4 AC → telemetry only.
- Fidelity findings: setups only separate near the limit (> ~1.5 g); static camber only acts with MF 5.2; dampers change transients not steady state; roll-centre height halves the ARB effect.
- Reference: TUM QSS laptime-sim F1_Shanghai = **91.24 s** (different car/tyres from ours).

---

## 7. Implementation state (`ocd_first_lap`)

Path: `/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap`. Read its README.md — it is authoritative. Key pieces: `ocd_lap/backend.py` (VehicleBackend, OCDBackend), `ocd_lap/env.py` (OCDRacingEnv skeleton, Stage 5), `ocd_lap/track.py`, `ocd_lap/car_setup.py`, `ocd_lap/qss_lap.py`, `scripts/qa_gate.sh`.

Known problems: OCD not yet fully validated for RL (Stage 4: 3 FAIL); max brake torque default **360 N·m is far too low**; diff coast lock no effect; aero map placeholders; no elevation; car = OCD default chassis/tyres with F1 mass/geometry.

**Rule:** verify against the actual code before asserting what OCD does or lacks.

---

## 8. Generalization protocol (setup agent only)

Hold one axis out (car / driver / track). Metrics: zero-shot regret vs CMA-ES/BO, warm- vs cold-start runs-to-converge, critic rank-correlation on held-out setups.

---

## 9. Timeline and resources

~1.5 months implementation + ~0.5 month writing (as of early Oct 2026). Time is the binding constraint: favour the cheaper path, flag effort honestly. Hardware: M4 Mac, L5 GPU, lab PCs, CarSim on one lab PC.

---

## 10. Literature (driving-agent relevant subset)

Haarnoja et al. 2018 (SAC); Fujimoto et al. 2018 (TD3); Ng et al. 1999 (reward shaping); Linesight (Trackmania RL); Fuchs et al. (GT Sport RL); GT Sophy (Wurman et al., Nature 2022); Remonda et al. AC Gym (NeurIPS 2024); Subosits et al. arXiv:2412.03803 (setup-conditioned RL driver, TRI); Song et al. Science Robotics 2023 (RL vs OC); Hoeppke 2022 Oxford PhD; Leyton-Romero & Limebeer VSD 2026; Heilmeier et al. VSD 2020 (min-curvature line).

---

## 11. OPEN decisions (ask, don't assume)

1. 26 vs 16 setup knobs. 2. AC now vs deferred. 3. Keep RL driving agent vs AC AI driver. 4. How to validate OCD suspension features. 5. Leyton-Romero & Limebeer as formal baseline. 6. SF70H aero/engine data; **brake torque value**; diff/engine braking. 7. Track-summary thresholds, roughness. 8. Training-schedule details (K fine-tune steps, B laps, N, λ, c).

---

## 12. Superseded — do not reintroduce

Driver style vector z as input (driver is only ξref); corridor/smoothness driver attributes; problem_formulation_v5 (time-optimal driver that sees the setup, c annealed to 0, Speed Dreams); raw track points / full ξref into the setup agent; Speed Dreams/CARLA/F1TENTH; RunPod; RL on PS4; multi-agent RL framing; OCP/NLP as the solver; GT-car scope; "2014 OC superseded by RL" framing.

---

## 13. How to work with Ibrahim

- **Verify, don't recall.** Run the code, read the source, cite the paper. Say which statements come from a paper vs your interpretation.
- **Plain language first.** Core idea before algebra. Visual learner: dark-mode, clean diagrams.
- **Formulation is collaborative.** Surface decisions as direct questions in chat.
- **Edit, don't rewrite** his writing.
- Docs: LaTeX, IEEE-conference look. Emails to Prof. Naseem: short, lists not tables.
- Casual and fast; be direct, concise, honest about uncertainty and effort. Commit after each stage.
