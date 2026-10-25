from rlsetup.provenance import run_provenance


def test_run_provenance_fields():
    p = run_provenance()
    for repo in ('rlsetup', 'ocd_first_lap'):
        assert len(p[repo]['commit']) == 40
        assert isinstance(p[repo]['dirty'], bool) and isinstance(p[repo]['untracked'], bool)
    assert len(p['ocd_lap_code_version']) == 12
