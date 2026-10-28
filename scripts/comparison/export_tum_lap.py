#!/usr/bin/env python3
"""Export TUM laptime-simulation's own Shanghai lap, per raceline point, for three variants.

Runs in TUM's venv, not this repo's (TUM needs NumPy 1.x); setup as in the docstring of
ocd_first_lap/scripts/run_tum_laptimesim.py:
    git clone https://github.com/TUMFTM/laptime-simulation.git ~/Documents/tum/laptime-simulation
    git -C ~/Documents/tum/laptime-simulation checkout afa0aa710adb9fdd23fa3d6fbdd6a13664253616
    python3.12 -m venv ~/Documents/tum/tumenv
    ~/Documents/tum/tumenv/bin/pip install "numpy==1.26.4" "scipy<1.14" "matplotlib<3.9" quadprog
    ~/Documents/tum/tumenv/bin/pip install --no-deps "trajectory-planning-helpers==0.66"
    ~/Documents/tum/tumenv/bin/python scripts/comparison/export_tum_lap.py ~/Documents/tum/laptime-simulation

The Track / CarHybrid / Driver / Lap construction and every option (TRACK, SOLVER, DRIVER, DEBUG,
CASES) are imported from ocd_first_lap's run_tum_laptimesim.py (read-only), so the settings are
exactly those of main_laptimesim.py with only the ERS/DRS switches changed. Variants exported:
    tum_default  DRS on, ERS FCFB 4 MJ, recuperation on   (main_laptimesim.py as shipped; 91.24 s)
    ers          DRS off, ERS FCFB 4 MJ, no recuperation  (closest to OCD sf70h_tum with ERS on)
    none         DRS off, no ERS
Writes comparison/tum/tum_<variant>.csv + .meta.json (columns: common.TUM_COLUMNS).
Units SI; s along TUM's 5 m re-interpolated raceline; one row per raceline point plus the
closing point.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

VARIANTS = ('tum_default', 'ers', 'none')
EXPECTED_TUM_DEFAULT_S = 91.238   # docs/sf70h/tum_laptimesim.json in ocd_first_lap (91.24 s)


def load_tum_runner(ocd_first_lap):
    path = Path(ocd_first_lap) / 'scripts' / 'run_tum_laptimesim.py'
    spec = importlib.util.spec_from_file_location('run_tum_laptimesim', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def simulate(repo, runner, variant):
    """Build TUM's objects exactly as run_tum_laptimesim.run() and return (track, car, driver, lap)."""
    import laptimesim
    drs, changes = runner.CASES[variant]
    track = laptimesim.src.track.Track(
        pars_track=dict(runner.TRACK, use_drs1=drs, use_drs2=drs),
        parfilepath=str(repo / 'laptimesim/input/tracks/track_pars.ini'),
        trackfilepath=str(repo / 'laptimesim/input/tracks/racelines/Shanghai.csv'),
        vel_lim_glob=np.inf)
    car = laptimesim.src.car_hybrid.CarHybrid(parfilepath=str(repo / 'laptimesim/input/vehicles/F1_Shanghai.ini'))
    driver = laptimesim.src.driver.Driver(carobj=car, pars_driver=dict(runner.DRIVER, **changes),
                                          trackobj=track, stepsize=track.stepsize)
    lap = laptimesim.src.lap.Lap(driverobj=driver, trackobj=track, pars_solver=dict(runner.SOLVER),
                                 debug_opts=dict(runner.DEBUG))
    lap.simulate_lap()
    return track, car, driver, lap


def rows_of(track, car, driver, lap):
    """One dict per raceline point (N unclosed points + the closing point)."""
    n = track.no_points
    v, s = lap.vel_cl, track.dists_cl
    ax = np.diff(v ** 2) / (2 * np.diff(s))          # step i -> i+1
    tires = car.pars_tires
    rows = []
    for i in range(n + 1):
        closing = i == n
        j = 0 if closing else i                        # geometry of the closing point = point 0
        r = dict(i=i, closing_point=int(closing), s_m=s[i], x_m=track.raceline[j, 0], y_m=track.raceline[j, 1],
                 kappa_radpm=track.kappa[j], t_s=lap.t_cl[i], speed_mps=v[i], ay_mps2=v[i] ** 2 * track.kappa[j],
                 gear=int(lap.gear_cl[i]), engine_rpm=lap.n_cl[i] * 60., ers_energy_J=lap.es_cl[i],
                 ers_used_J=lap.e_cons_cl[i], fuel_used_kg=lap.fuel_cons_cl[i], mu=track.mu[j])
        if not closing:
            n_rps = lap.n_cl[i]
            r.update(ax_mps2=ax[i], engine_torque_crank_Nm=lap.m_eng[i], e_motor_torque_crank_Nm=lap.m_e_motor[i],
                     requested_torque_crank_Nm=lap.m_requ[i], engine_power_W=2 * math.pi * n_rps * lap.m_eng[i],
                     e_motor_power_W=2 * math.pi * n_rps * lap.m_e_motor[i],
                     ers_recuperated_step_J=lap.e_rec_e_motor[i], drs=int(bool(track.drs[i])),
                     em_boost_use=int(bool(driver.em_boost_use[i])), throttle_pos=driver.throttle_pos[i])
            for k, w in enumerate(common.WHEELS):
                fz = lap.tire_loads[i, k]
                t = tires['f' if w[0] == 'F' else 'r']
                r[f'Fz_{w}_N'] = fz
                r[f'fx_pot_{w}_N'] = track.mu[i] * (t['mux'] + t['dmux_dfz'] * (fz - t['fz_0'])) * fz
                r[f'fy_pot_{w}_N'] = track.mu[i] * (t['muy'] + t['dmuy_dfz'] * (fz - t['fz_0'])) * fz
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tum_repo', help='clone of TUMFTM/laptime-simulation at ' + common.TUM_COMMIT)
    ap.add_argument('--ocd-first-lap', default='/Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap')
    ap.add_argument('--out', default=str(common.OUT_ROOT / 'tum'))
    args = ap.parse_args()
    # Same NumPy-1.24 alias shim as run_tum_laptimesim.main() (tph 0.66 uses np.int etc.).
    for alias, typ in (('int', int), ('float', float), ('bool', bool)):
        if alias not in np.__dict__:
            setattr(np, alias, typ)
    repo = Path(args.tum_repo).expanduser().resolve()
    tum_git = common.git_state(repo)
    if tum_git['commit'] != common.TUM_COMMIT or tum_git['dirty']:
        raise SystemExit(f'TUM clone must be clean at {common.TUM_COMMIT}: {tum_git}')
    sys.path.insert(0, str(repo))
    runner, runner_path = load_tum_runner(args.ocd_first_lap)
    inputs = {k: repo / p for k, p in {
        'vehicle_ini': 'laptimesim/input/vehicles/F1_Shanghai.ini',
        'track_pars_ini': 'laptimesim/input/tracks/track_pars.ini',
        'raceline_csv': 'laptimesim/input/tracks/racelines/Shanghai.csv'}.items()}
    vendored = Path(args.ocd_first_lap) / 'vendor/racetrack-database/racelines/Shanghai.csv'
    import importlib.metadata as md
    for variant in VARIANTS:
        track, car, driver, lap = simulate(repo, runner, variant)
        rows = rows_of(track, car, driver, lap)
        path = common.write_csv(Path(args.out) / f'tum_{variant}.csv', 'tum', rows)
        lap_time = float(lap.t_cl[-1])
        drs, changes = runner.CASES[variant]
        if variant == 'tum_default' and abs(lap_time - EXPECTED_TUM_DEFAULT_S) > 5e-3:
            raise SystemExit(f'tum_default lap {lap_time:.4f} s does not reproduce {EXPECTED_TUM_DEFAULT_S} s')
        common.write_meta(path, {
            'source': 'tum', 'variant': variant, 'lap_time_s': lap_time,
            'v_max_kmh': float(3.6 * lap.vel_cl.max()), 'v_min_kmh': float(3.6 * lap.vel_cl.min()),
            'raceline_length_m': float(track.dists_cl[-1]), 'stepsize_m': float(track.stepsize),
            'n_points': int(track.no_points), 'ers_used_J': float(lap.e_cons_cl[-1]),
            'ers_final_J': float(lap.es_cl[-1]), 'mguk_recuperated_J': float(np.sum(lap.e_rec_e_motor)),
            'drs_share_of_points': float(np.mean(track.drs)),
            'settings': {'drs': drs, 'driver_changes': changes, 'track': dict(runner.TRACK, use_drs1=drs, use_drs2=drs),
                         'solver': runner.SOLVER, 'driver': dict(runner.DRIVER, **changes), 'debug': runner.DEBUG},
            'model': 'TUM laptime-simulation quasi-steady-state forward/backward solver (FB+), point mass with '
                     'per-wheel load transfer, axle-wise friction check, F1_Shanghai.ini',
            'code': {'tum_laptime_simulation': {'url': common.TUM_REPO_URL, **tum_git},
                     'run_tum_laptimesim_py': {'path': str(runner_path), 'sha256': common.sha256_file(runner_path)},
                     'ocd_first_lap': common.git_state(args.ocd_first_lap),
                     'this_repo': common.this_repo_state(),
                     'script': {'path': 'scripts/comparison/export_tum_lap.py', 'sha256': common.sha256_file(__file__),
                                'common_sha256': common.sha256_file(common.__file__)}},
            'inputs_sha256': {**{k: common.sha256_file(p) for k, p in inputs.items()},
                              'ocd_first_lap_vendored_raceline': common.sha256_file(vendored)},
            'python': sys.version.split()[0],
            'packages': {p: md.version(p) for p in ('numpy', 'scipy', 'matplotlib', 'trajectory-planning-helpers', 'quadprog')},
            'command': ' '.join(sys.argv)})
        print(f'{variant:12s} {lap_time:8.3f} s  v_max {3.6 * lap.vel_cl.max():6.1f} km/h  '
              f'ERS used {lap.e_cons_cl[-1] / 1e6:.2f} MJ  -> {path}', flush=True)


if __name__ == '__main__':
    main()
