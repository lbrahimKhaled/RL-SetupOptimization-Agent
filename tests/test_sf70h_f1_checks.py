"""Pure helpers of scripts/sf70h_f1_checks.py (no ocd_first_lap sf70h-f1 checkout needed)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('sf70h_f1_checks', ROOT / 'scripts' / 'sf70h_f1_checks.py')
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)


def test_ggv_row_valid():
    ok = dict(speed_mps=90., v_traction=110., ax_traction=30., v_brake=70., ax_brake=45.)
    assert chk.ggv_row_valid(ok)
    assert not chk.ggv_row_valid(dict(ok, v_traction=46.))          # settle collapsed
    assert not chk.ggv_row_valid(dict(ok, ax_traction=0.))
    assert not chk.ggv_row_valid(dict(ok, v_brake=16., ax_brake=290.))
    assert not chk.ggv_row_valid(dict(ok, ax_brake=80.1))


def test_update_merges_keys_and_subkeys(tmp_path):
    path = tmp_path / 'r.json'
    chk.update('a', {'x': np.float64(1.5), 'y': np.arange(2)}, path=path)
    chk.update('ocp', {'lap': 1.}, path=path, sub='sha')
    chk.update('ocp', {'lap': 2.}, path=path, sub='bcn')
    chk.update('b', float('nan'), path=path)
    d = json.loads(path.read_text())
    assert d == {'a': {'x': 1.5, 'y': [0, 1]}, 'ocp': {'sha': {'lap': 1.}, 'bcn': {'lap': 2.}}, 'b': None}


def test_part_closed_loop_reads_run_folders(tmp_path):
    d = tmp_path / 'lbl'
    d.mkdir()
    run = dict(f=1.0, termination='lap2_complete', lap_times_s=[103., 102.5], lap2_s=102.5, spin_at_s_raceline_m=None,
               spin_lap=None, on_track_cog_fraction_lap2=.95, on_track_footprint_fraction_lap2=.7,
               max_abs_e_raceline_lap2_m=9., rms_e_raceline_lap2_m=2., v_max_lap2_kmh=280.)
    (d / 'runs.json').write_text(json.dumps(dict(runs=[run], setup={'a': 1}, gains={'tc_slip': .06}, command='x')))
    (d / 'analysis.json').write_text(json.dumps({'1.00': {'loss_by_limit_s': {'traction': 3.}}}))
    (d / 'cl_f1.00.meta.json').write_text(json.dumps({'conditions': {'car': 'sf70h_f1', 'setup_class': 'SF70HF1Setup'}}))
    out = chk.part_closed_loop(tmp_path)
    row = out['runs']['lbl']
    assert row['car'] == 'sf70h_f1' and row['gains'] == {'tc_slip': .06}
    assert row['per_f']['1.00']['lap2_s'] == 102.5 and row['per_f']['1.00']['loss_by_limit_s'] == {'traction': 3.}
    assert out['reference_debug_car_lap2_s'] == {'1.00': 103.68, '0.96': 102.985}
