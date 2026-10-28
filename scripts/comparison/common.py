"""Shared pieces of the comparison exports (TUM lap, pure pursuit in OCD, RL driver).

Imports only the standard library and numpy, because `export_tum_lap.py` runs in TUM's own venv
(numpy 1.26) and the other two in this repo's venv (numpy 2.x).

`SCHEMAS[source]` is the column list (name, unit, meaning) of every CSV of that source; the
README tables are rendered from it (`python scripts/comparison/common.py`) and
tests/test_comparison_schema.py checks that the exported headers and the README agree with it.
CSV format: comma separated, header row, one row per sample, SI units, floats with 7 significant
digits, integers and 0/1 flags as integers, missing values as empty fields.
"""
from pathlib import Path
import datetime
import hashlib
import json
import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / 'comparison'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from rlsetup.provenance import git_state, sha256_file   # noqa: E402  (stdlib only)

TUM_REPO_URL = 'https://github.com/TUMFTM/laptime-simulation'
TUM_COMMIT = 'afa0aa710adb9fdd23fa3d6fbdd6a13664253616'
WHEELS = ('FL', 'FR', 'RL', 'RR')

# --- column schemas ---------------------------------------------------------------------------------
_W = lambda fmt, unit, text: [(fmt.format(w=w), unit, text.format(w=w)) for w in WHEELS]

TUM_COLUMNS = [
    ('i', '-', 'raceline point index after TUM\'s 5 m re-interpolation (0 = start/finish line); the last row is the closing point N = first point of the next lap'),
    ('closing_point', '0/1', '1 on the closing row; per-step columns are empty there'),
    ('s_m', 'm', 'arc length along TUM\'s interpolated raceline (Track.dists_cl, spline length)'),
    ('x_m', 'm', 'raceline point x (track frame of the racetrack-database, same as ocd_lap Track.reference)'),
    ('y_m', 'm', 'raceline point y'),
    ('kappa_radpm', '1/m', 'curvature of the raceline, filtered as in TUM (Track.kappa); positive = turning left'),
    ('t_s', 's', 'time at this point since the line (Lap.t_cl); the closing row holds the lap time'),
    ('speed_mps', 'm/s', 'speed at this point (Lap.vel_cl)'),
    ('ax_mps2', 'm/s^2', 'longitudinal acceleration over the step i -> i+1: (v[i+1]^2 - v[i]^2) / (2 ds) (point-mass kinematics)'),
    ('ay_mps2', 'm/s^2', 'lateral acceleration v^2 kappa at this point; positive = left'),
    ('gear', '-', 'gear, 1-based (Lap.gear_cl)'),
    ('engine_rpm', '1/min', 'engine speed (Lap.n_cl x 60)'),
    ('engine_torque_crank_Nm', 'N m', 'combustion-engine torque used in the step (Lap.m_eng, crankshaft)'),
    ('e_motor_torque_crank_Nm', 'N m', 'electric-motor (MGU-K) torque used in the step (Lap.m_e_motor, crankshaft)'),
    ('requested_torque_crank_Nm', 'N m', 'powertrain torque the tyres could transmit in the step (Lap.m_requ, crankshaft)'),
    ('engine_power_W', 'W', '2 pi n m_eng'),
    ('e_motor_power_W', 'W', '2 pi n m_e_motor (mechanical)'),
    ('ers_energy_J', 'J', 'energy-storage state at this point (Lap.es_cl); 0 for variant none'),
    ('ers_used_J', 'J', 'electric energy drawn by the motor since the line, incl. efficiency (Lap.e_cons_cl)'),
    ('ers_recuperated_step_J', 'J', 'MGU-K recuperation in the step (Lap.e_rec_e_motor; turbocharger harvest is not stored by TUM)'),
    ('fuel_used_kg', 'kg', 'fuel used since the line (Lap.fuel_cons_cl)'),
    ('drs', '0/1', 'DRS open in this step (Track.drs; drag x (1 - drs_factor) while accelerating)'),
    ('em_boost_use', '0/1', 'driver allows the e-motor boost in this step (Driver.em_boost_use)'),
    ('throttle_pos', '-', 'driver throttle position in this step (Driver.throttle_pos; 1 unless lift-and-coast / yellow flags)'),
    ('mu', '-', 'track friction factor (Track.mu = mu_mean x mu_weather)'),
    *_W('Fz_{w}_N', 'N', 'tyre load {w} as stored by the solver (Lap.tire_loads: static + load transfer + aero)'),
    *_W('fx_pot_{w}_N', 'N', 'longitudinal force potential {w}: mu (mux + dmux_dfz (Fz - fz_0)) Fz, TUM car.tire_force_pots law on Fz_{w}_N'),
    *_W('fy_pot_{w}_N', 'N', 'lateral force potential {w}: mu (muy + dmuy_dfz (Fz - fz_0)) Fz, same law'),
]

OCD_COLUMNS = [
    ('t_s', 's', 'time since the start of the recording (car placed 200 m before the line; RL: since hand-over after the 2 s settle)'),
    ('lap', '-', 'line crossings before this sample: 0 = run-in, 1 = lap 1, 2 = lap 2 (the flying lap), 3 = after lap 2'),
    ('t_lap_s', 's', 'time since the last line crossing (crossing interpolated between ticks); time since start on the run-in'),
    ('progress_m', 'm', 'distance driven along Track.path since the start (unwrapped)'),
    ('s_path_m', 'm', 'arc length on Track.path (ocd_first_lap safe minimum-curvature line), [0, 5388.77)'),
    ('s_raceline_m', 'm', 'arc length on Track.reference (TUM raceline polyline), [0, 5340.77)'),
    ('x_m', 'm', 'CoG x (track frame)'),
    ('y_m', 'm', 'CoG y'),
    ('yaw_rad', 'rad', 'yaw angle (OCD psi, counter-clockwise from +x; not wrapped)'),
    ('speed_mps', 'm/s', 'hypot(vx, vy)'),
    ('vx_mps', 'm/s', 'body-frame longitudinal velocity'),
    ('vy_mps', 'm/s', 'body-frame lateral velocity, positive = left'),
    ('yaw_rate_radps', 'rad/s', 'yaw rate, positive = counter-clockwise (left)'),
    ('ax_mps2', 'm/s^2', 'body-frame acceleration dvx/dt - r vy from OCD\'s state derivative at the sample'),
    ('ay_mps2', 'm/s^2', 'body-frame acceleration dvy/dt + r vx from OCD\'s state derivative, positive = left'),
    ('steer_cmd_rad', 'rad', 'steering command (OCD steering_actuator target), positive = left'),
    ('steer_actual_rad', 'rad', 'actual steering angle after OCD\'s PT1 actuator (0.6 rad, 1.5 rad/s limits)'),
    ('throttle', '-', 'throttle pedal [0, 1]'),
    ('brake', '-', 'brake pedal [0, 1] (x 12 kN m total, split by the setup brake bias 0.5)'),
    ('drive_torque_axle_Nm', 'N m', 'rear-axle drive torque OCD received in the last 2 ms step (transmission output)'),
    *_W('brake_torque_{w}_Nm', 'N m', 'brake torque {w} OCD received in the last 2 ms step'),
    ('gear', '-', 'gear, 1-based (ocd_lap.powertrain, TUM F1 ratios)'),
    ('engine_rpm', '1/min', 'engine speed = rear-axle speed x overall ratio of the current gear'),
    ('engine_torque_crank_Nm', 'N m', 'throttle x full-load engine torque at engine_rpm (ocd_lap.powertrain, recomputed at the sample)'),
    ('e_motor_torque_crank_Nm', 'N m', 'throttle x e-motor torque at engine_rpm and speed (0 with ERS off or empty store; recomputed at the sample)'),
    ('ers_energy_J', 'J', 'ERS energy left (refilled to 4 MJ at each line crossing); empty with ERS off'),
    ('slip_rl', '-', 'rear-left longitudinal slip ratio (backend state)'),
    ('slip_rr', '-', 'rear-right longitudinal slip ratio (backend state)'),
    *_W('kappa_{w}', '-', 'longitudinal slip ratio {w} (OCD x_vec)'),
    *_W('alpha_{w}_rad', 'rad', 'slip angle {w} (OCD x_vec)'),
    *_W('Fz_{w}_N', 'N', 'vertical tyre force {w} (OCD)'),
    ('e_raceline_m', 'm', 'lateral offset to TUM\'s raceline (Track.reference), positive = left'),
    ('heading_err_raceline_rad', 'rad', 'yaw minus raceline heading, wrapped to [-pi, pi)'),
    ('e_path_m', 'm', 'lateral offset to Track.path (the RL driver\'s safe line xi_ref), positive = left'),
    ('track_position', '-', 'lateral position across the track at s_path: -1 right edge ... +1 left edge (DrivingEnv.track_position)'),
    ('on_track', '0/1', 'car footprint (4.6 x 2.0 m box) inside the track at every 2 ms physics step of the tick ending here (Track.contains_car, the env\'s own check)'),
]

PP_COLUMNS = OCD_COLUMNS + [
    ('target_speed_mps', 'm/s', 'speed-controller target: f x v_TUM(ers) at the raceline point 0.5 s x vx ahead'),
]

RL_COLUMNS = OCD_COLUMNS + [
    ('decision', '-', 'index of the 25 Hz decision this tick belongs to (0 = first decision after hand-over)'),
    ('decision_end', '0/1', '1 on the tick that ends a decision; the columns from `reward` on are filled only there (stochastic files hold only these rows)'),
    ('a_steer', '-', 'raw policy action held over the decision, steering, [-1, 1] (x 0.6 rad)'),
    ('a_throttle', '-', 'raw policy action, throttle, [-1, 1] (pedal = clip((a + 0.5) / 1.5, 0, 1))'),
    ('a_brake', '-', 'raw policy action, brake, [-1, 1] (same deadzone mapping)'),
    ('reward', '-', 'reward of the decision (time_pbrs: r_progress - r_time - r_penalty + r_shaping)'),
    ('r_progress', '-', 'a ds over the decision (a = 1 per m of Track.path progress)'),
    ('r_time', '-', 'b dt over the decision (b = 1 per s)'),
    ('r_smoothness', '-', 'b dt du^T W du (logged only, not rewarded in time_pbrs)'),
    ('r_tracking', '-', 'c e^2 to the safe line (logged only, not rewarded in time_pbrs)'),
    ('r_penalty', '-', 'failure penalty P (100 for off track / non-finite / wrong way)'),
    ('r_shaping', '-', 'potential-based shaping gamma Phi(s\') - Phi(s), Phi = -|e_path|'),
    ('e_ref_env_m', 'm', 'the env\'s own offset to the safe line at the decision end (info e_ref; equals e_path_m)'),
    ('termination', 'text', 'env termination at this decision: off_track, nonfinite, wrong_way, lap_complete (empty otherwise)'),
    ('truncation', 'text', 'env truncation at this decision: time_limit, stall (empty otherwise)'),
]

SCHEMAS = {'tum': TUM_COLUMNS, 'tum_pure_pursuit_ocd': PP_COLUMNS, 'rl': RL_COLUMNS}


def names(source):
    return [c[0] for c in SCHEMAS[source]]


def markdown_table(source):
    rows = ['| Column | Unit | Meaning |', '|---|---|---|']
    rows += [f'| `{n}` | {u} | {m} |' for n, u, m in SCHEMAS[source]]
    return '\n'.join(rows)


# --- writing ---------------------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    if isinstance(v, (bool,)) or type(v).__name__ == 'bool_':
        return '1' if v else '0'
    if isinstance(v, int) or type(v).__name__.startswith('int'):
        return str(int(v))
    v = float(v)
    return '' if not math.isfinite(v) else f'{v:.7g}'


def write_csv(path, source, rows):
    """rows: list of dicts with exactly the schema's columns (missing keys -> empty)."""
    cols = names(source)
    extra = set().union(*(r.keys() for r in rows)) - set(cols) if rows else set()
    if extra:
        raise KeyError(f'{path}: columns not in the {source} schema: {sorted(extra)}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(_fmt(r.get(c)) for c in cols) + '\n')
    return path


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, 'tolist'):
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def write_meta(csv_path, meta):
    """<csv>.meta.json next to the CSV; adds the CSV's sha256, row count and the UTC time."""
    csv_path = Path(csv_path)
    with csv_path.open() as f:
        n = sum(1 for _ in f) - 1
    meta = dict(meta, file=csv_path.name, rows=n, sha256=sha256_file(csv_path),
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    p = csv_path.with_suffix('.meta.json')
    p.write_text(json.dumps(_clean(meta), indent=1, allow_nan=False) + '\n')
    return p


def this_repo_state():
    return git_state(REPO_ROOT)


def script_sha256(path):
    return sha256_file(path)


if __name__ == '__main__':   # print the README column tables
    for src in SCHEMAS:
        print(f'### {src}\n\n{markdown_table(src)}\n')
