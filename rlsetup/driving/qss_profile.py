"""Quasi-steady-state speed profile v(s) [m/s] on Track.path for one setup (p0 by default).

`ocd_lap.qss_lap.lap_time` returns scalars only, so the profile is recomputed here from its
public pieces (`measure_ggv`, `machine_limit`, TUM `calc_vel_profile`), replicating
qss_lap.py lines 94-108 exactly; tests check that `calc_t_profile` of this profile equals
`qss_lap.lap_time(..., ggv_data=same)['lap_time_s']`.

Used only for the start-speed distribution of resets and the demo driver's target speed. It is
computed once for p0 and used for every setup (review R5: same start distribution per setup).
Cached as data/qss_profile_<track>_p0.npz with its inputs (cache key) and provenance. A cache
whose key does not match the current code/track is recomputed in memory with a warning and is
never rewritten silently (pass `rebuild=True`, or run scripts/build_qss_profile.py).
"""
from dataclasses import dataclass
from pathlib import Path
import datetime
import hashlib
import json
import warnings
import numpy as np
import trajectory_planning_helpers as tph
import ocd_native
from ocd_lap import qss_lap
from ocd_lap.backend import OCDBackend, StartPose, CONFIG
from ocd_lap.car_setup import Setup
from ocd_lap.translator import P
from ..provenance import REPO_ROOT, run_provenance

DATA_DIR = REPO_ROOT / 'data'
BRAKE_TORQUE_NM = 12000.   # total brake torque of the QSS brake ramp (qss_lap default, EnvConfig default)
# ocd_lap modules whose code determines the profile (the ocd_lap code_version omits some).
# rig.py and vertical_model.py: imported by handling.py (D2 group 0, lead decision).
_SOURCES = ('qss_lap.py', 'handling.py', 'manoeuvres.py', 'powertrain.py', 'backend.py',
            'translator.py', 'car_setup.py', 'aero_map.py', 'rig.py', 'vertical_model.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(track, setup, car='av21', ers=False, brake_torque_Nm=BRAKE_TORQUE_NM):
    """Everything the profile depends on; a cache is valid only if this matches exactly.
    The AV-21 key has no car fields and a 12 kN m key no brake field, so caches built before the
    car / brake options stay valid."""
    lap_dir = Path(qss_lap.__file__).resolve().parent
    extra = {} if car == 'av21' else {'car': car, 'ers': bool(ers)}
    if brake_torque_Nm != BRAKE_TORQUE_NM:
        extra['brake_torque_Nm'] = float(brake_torque_Nm)
    return json.loads(json.dumps({**extra,
        'schema': 1, 'track': track.name,
        'track_path_sha256': hashlib.sha256(np.ascontiguousarray(track.path).tobytes()).hexdigest(),
        'setup': setup.to_dict(),
        'ocd_lap_sources_sha256': {f: _sha(lap_dir / f) for f in _SOURCES},
        'ocd_native_sha256': _sha(ocd_native.__file__),
        'ocd_config_sha256': _sha(CONFIG),
    }))  # JSON round trip, so it compares equal to a key loaded from the cache


@dataclass
class QSSProfile:
    s: np.ndarray            # Track.path node arc lengths [m], closed (last = lap length)
    v: np.ndarray            # QSS speed at each node [m/s], closed (last = first)
    lap_time_s: float
    meta: dict

    @property
    def length(self):
        return float(self.s[-1])

    def speed(self, s):
        """v_qss at path arc length(s) s [m] (wraps)."""
        return np.interp(np.asarray(s, dtype=float) % self.length, self.s, self.v)

    def min_speed(self, s0, distance_m, step_m=1.):
        """Minimum of v_qss over [s0, s0 + distance_m] (sampled every step_m and at both ends)."""
        n = max(int(np.ceil(distance_m / step_m)), 1)
        return float(self.speed(s0 + np.linspace(0., distance_m, n + 1)).min())


def compute(track, setup=None, *, ggv_data=None, car='av21', ers=False, brake_torque_Nm=BRAKE_TORQUE_NM):
    """Profile for `setup` (default p0 = Setup()) on track.path: (QSSProfile, ggv_data).
    Other cars than 'av21' take `qss_lap.lap_time`'s own profile (its car geometry, tyres and
    friction-ellipse exponent; DRS off, as OCDBackend has none). `brake_torque_Nm`: total torque
    of the g-g-v brake ramp (the env's max_brake_torque_Nm)."""
    setup = Setup() if setup is None else setup
    if ggv_data is None and brake_torque_Nm != BRAKE_TORQUE_NM:
        ggv_data = qss_lap.measure_ggv(setup, car=car, brake_torque_Nm=brake_torque_Nm)
    if car != 'av21':
        r = qss_lap.lap_time(setup, track, ggv_data=ggv_data, car=car, ers=ers, drs=False)
        v = np.asarray(r['v_profile'], dtype=float)
        meta = {'key': cache_key(track, setup, car, ers, brake_torque_Nm), 'ggv': r['ggv'], 'traction': r['traction'],
                'drag_coeff': r['drag_coeff']}
        profile = QSSProfile(np.array(track.path[:, 0]), np.r_[v, v[0]], r['lap_time_s'], meta)
        return profile, (np.asarray(r['ggv']), np.asarray(r['traction']), r['details'])
    ggv, traction, details = ggv_data or qss_lap.measure_ggv(setup)
    # --- qss_lap.lap_time, lines 95-109 (same calls, same arguments) ---
    backend = OCDBackend()
    backend.reset(setup, StartPose(speed=0.))
    params = backend.params
    v_grid = np.arange(0., 101., 5.)
    engine = qss_lap.machine_limit(setup, params, v_grid)
    trac = np.interp(v_grid, traction[:, 0], traction[:, 1])
    ax_machines = np.column_stack([v_grid, np.minimum(engine, trac)])
    ggv_full = np.vstack([[0., ggv[0, 1], ggv[0, 2]], ggv, [100., ggv[-1, 1], ggv[-1, 2]]])
    drag = .5 * params[P + 'aerodynamics.air_density_kgpm3'] * params[P + 'aerodynamics.c_d'] \
        * params[P + 'aerodynamics.A_m2']
    el = np.diff(track.path[:, 0])
    v = tph.calc_vel_profile.calc_vel_profile(
        ax_max_machines=ax_machines, kappa=track.curvature, el_lengths=el, closed=True,
        drag_coeff=drag, m_veh=params[P + 'mass_vehicle_kg'], ggv=ggv_full, v_max=100.)
    t = tph.calc_t_profile.calc_t_profile(vx_profile=np.r_[v, v[0]], el_lengths=el)
    # ---
    meta = {'key': cache_key(track, setup, brake_torque_Nm=brake_torque_Nm), 'ggv': np.asarray(ggv).tolist(),
            'traction': np.asarray(traction).tolist(), 'drag_coeff': drag}
    profile = QSSProfile(np.array(track.path[:, 0]), np.r_[v, v[0]], float(t[-1]), meta)
    return profile, (ggv, traction, details)


def default_path(track_name, tag='p0', car='av21', ers=False, brake_torque_Nm=BRAKE_TORQUE_NM):
    suffix = '' if car == 'av21' else f'_{car}' + ('_ers' if ers else '')
    if brake_torque_Nm != BRAKE_TORQUE_NM:
        suffix += f'_brake{brake_torque_Nm:g}'
    return DATA_DIR / f'qss_profile_{track_name}_{tag}{suffix}.npz'


def save(profile, path):
    meta = dict(profile.meta, lap_time_s=profile.lap_time_s,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                provenance=run_provenance())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, s=profile.s, v=profile.v, lap_time_s=profile.lap_time_s, meta=json.dumps(meta))


def load(path):
    with np.load(path) as f:
        return QSSProfile(f['s'], f['v'], float(f['lap_time_s']), json.loads(str(f['meta'])))


def load_or_build(track, path=None, *, rebuild=False, car='av21', ers=False, setup='p0',
                  brake_torque_Nm=BRAKE_TORQUE_NM):
    """Profile of the named setup `setup` (rlsetup.driving.setups) for `track`: from the cache
    file if its key matches, else recomputed. Writes the file only if it is missing or `rebuild`.
    Call once in the main process before starting vector-env workers."""
    from .setups import get as get_setup
    path = Path(path) if path is not None else default_path(track.name, setup, car=car, ers=ers,
                                                            brake_torque_Nm=brake_torque_Nm)
    s = get_setup(setup)
    key = cache_key(track, s, car, ers, brake_torque_Nm)
    if path.exists() and not rebuild:
        cached = load(path)
        if cached.meta.get('key') == key:
            return cached
        warnings.warn(f'{path.name}: cache key does not match the current code/track; recomputing '
                      'in memory (rebuild with scripts/build_qss_profile.py)', stacklevel=2)
        return compute(track, s, car=car, ers=ers, brake_torque_Nm=brake_torque_Nm)[0]
    profile = compute(track, s, car=car, ers=ers, brake_torque_Nm=brake_torque_Nm)[0]
    save(profile, path)
    return load(path)


def for_env(track, env_cfg, **kw):
    """load_or_build for an EnvConfig: its car, ERS, named setup and brake torque."""
    return load_or_build(track, car=env_cfg.car, ers=env_cfg.ers, setup=env_cfg.setup,
                         brake_torque_Nm=env_cfg.max_brake_torque_Nm, **kw)
