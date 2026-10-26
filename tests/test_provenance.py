import json
import sys
from rlsetup import provenance
from rlsetup.provenance import run_provenance, git_state, OCD_LAP_EXTRA_SOURCES
from rlsetup.driving.config import EnvConfig


def test_run_provenance_fields():
    p = run_provenance()
    for repo in ('rlsetup', 'ocd_first_lap'):
        assert len(p[repo]['commit']) == 40
        assert isinstance(p[repo]['dirty'], bool) and isinstance(p[repo]['untracked'], bool)
        assert len(p[repo]['diff_sha256']) == 64
    assert len(p['ocd_lap_code_version']) == 12
    src = p['rlsetup_sources_sha256']
    assert {'rlsetup/provenance.py', 'rlsetup/driving/env.py', 'rlsetup/driving/config.py'} <= set(src)
    assert all(len(h) == 64 for h in src.values())
    extra = p['ocd_lap_extra_sources_sha256']
    assert {'ocd_lap/env.py', 'ocd_lap/qss_lap.py', 'ocd_lap/handling.py', 'ocd_lap/manoeuvres.py'} <= set(extra)
    assert all(h is not None and len(h) == 64 for h in extra.values())
    json.dumps(p)


def test_git_state_outside_a_work_tree(tmp_path):
    assert git_state(tmp_path) == {'path': str(tmp_path), 'commit': None, 'dirty': None,
                                   'untracked': None, 'diff_sha256': None}


def test_diff_hash_follows_tracked_changes(tmp_path):
    """diff_sha256 changes with an uncommitted edit of a tracked file and is the empty-input
    hash on a clean tree."""
    import subprocess
    run = lambda *a: subprocess.run(['git', '-C', str(tmp_path), *a], check=True, capture_output=True)
    run('init', '-q')
    (tmp_path / 'a.txt').write_text('one\n')
    run('add', 'a.txt')
    run('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'init')
    clean = git_state(tmp_path)
    assert not clean['dirty'] and clean['diff_sha256'] == provenance.hashlib.sha256(b'').hexdigest()
    (tmp_path / 'a.txt').write_text('two\n')
    dirty = git_state(tmp_path)
    assert dirty['dirty'] and dirty['diff_sha256'] != clean['diff_sha256']


def test_every_imported_ocd_lap_module_is_hashed():
    """Each ocd_lap module loaded by rlsetup is in ocd_lap's code_version or in
    OCD_LAP_EXTRA_SOURCES, so run provenance identifies all ocd_lap code a run used."""
    import rlsetup.driving.env, rlsetup.driving.qss_profile, rlsetup.driving.demo_driver  # noqa: F401
    import rlsetup.driving.train, rlsetup.driving.demos, rlsetup.driving.evaluate  # noqa: F401
    import ocd_lap.track  # noqa: F401  (train.py imports it inside the pipeline)
    from ocd_lap.simulation import SOURCES
    covered = set(SOURCES) | {'ocd_lap/evaluate.py', 'ocd_lap/provenance.py'} \
        | {f'ocd_lap/{f}' for f in OCD_LAP_EXTRA_SOURCES}
    loaded = {'ocd_lap/' + m.__file__.split('/ocd_lap/')[-1]
              for name, m in list(sys.modules.items())
              if (name == 'ocd_lap' or name.startswith('ocd_lap.')) and getattr(m, '__file__', None)}
    assert loaded - covered == set()


def test_env_metadata_is_json_and_identifies_the_env(make_env):
    env = make_env()
    m = env.env_metadata()
    json.dumps(m)
    assert m['config'] == EnvConfig().to_dict()
    assert m['config']['obs']['steer_scale_rad'] == .6
    assert m['reference']['digest'] == env.ref.digest and m['reference']['name'] == env.ref.name
    assert m['qss']['cache_key'] == env.qss.meta['key']
    assert m['qss']['cache_key']['track'] == env.track.name
    assert len(m['track']['path_sha256']) == 64
    assert m['observation_size'] == 159
