"""Pure helpers of platforms/windows/ac/ac_supervisor.py (no AC, no Windows calls)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platforms' / 'windows' / 'ac'))
import ac_supervisor as sup  # noqa: E402

R6B = """[FUEL]
VALUE=1

[FRONT_BIAS]
VALUE=54

[MGUH_MODE]
VALUE=1

[MGUK_DELIVERY]
VALUE=0

[MGUK_RECOVERY]
VALUE=0

[ARB_FRONT]
VALUE=0
"""


def test_setup_expectation_reads_the_shared_memory_fields():
    assert sup.setup_expectation(R6B) == dict(fuel=1., brake_bias=.54, ers_recovery=0., ers_power=0.,
                                              ers_heat_charging=1.)


def test_setup_matches_reports_the_differing_values():
    want = sup.setup_expectation(R6B)
    r6b = dict(fuel=1., brake_bias=.54, engine_brake=9, ers_recovery=0, ers_power=0, ers_heat_charging=1)
    default = dict(r6b, fuel=55., ers_recovery=5, ers_power=1)   # AC's default SF70H, 25 Nov 2026
    assert sup.setup_matches(want, r6b) == []
    assert sup.setup_matches(want, default) == ['fuel', 'ers_recovery', 'ers_power']
    assert sup.setup_matches(want, {}) == list(want)


def test_list_rows_skips_folders_without_setups(tmp_path):
    for d in ('drift', 'generic', 'imola', 'ks_barcelona'):
        (tmp_path/d).mkdir()
    (tmp_path/'generic'/'last.ini').write_text(R6B)
    for name in ('r6b', 'A_first', 'zz'):
        (tmp_path/'ks_barcelona'/f'{name}.ini').write_text(R6B)
    (tmp_path/'ks_barcelona'/'r6b.sp').write_text('')
    assert sup.list_rows(tmp_path, 'ks_barcelona', 'r6b') == (1, 1)


def test_menu_visible_needs_the_red_sidebar():
    img = np.full((768, 1024, 3), 90, np.uint8)
    assert not sup.menu_visible(img)
    x0, x1, y0, y1 = sup.MENU_BAR
    img[y0:y1, x0:x1] = (210, 10, 10)
    assert sup.menu_visible(img)
    assert not sup.menu_visible(img[:300, :80])   # too small to hold the sidebar


def test_pin_grip_fixes_the_dynamic_track_and_keeps_the_rest():
    text = ('[RACE]\r\nTRACK=ks_barcelona\r\n[DYNAMIC_TRACK]\r\nSESSION_START=200\r\nSESSION_TRANSFER=200\r\n'
            'RANDOMNESS=200\r\nLAP_GAIN=132\r\n\r\n[REMOTE]\r\nACTIVE=0\r\n')
    pinned = sup.pin_grip(text)
    assert pinned == text.replace('SESSION_START=200', 'SESSION_START=100').replace(
        'RANDOMNESS=200', 'RANDOMNESS=0').replace('LAP_GAIN=132', 'LAP_GAIN=0')
    assert sup.pin_grip(pinned) == pinned


def test_pin_grip_adds_missing_keys_and_section():
    assert sup.pin_grip('[DYNAMIC_TRACK]\n[REMOTE]\n') == \
        '[DYNAMIC_TRACK]\nSESSION_START=100\nRANDOMNESS=0\nLAP_GAIN=0\n[REMOTE]\n'
    assert sup.pin_grip('[RACE]\nTRACK=x\n') == '[RACE]\nTRACK=x\n[DYNAMIC_TRACK]\nSESSION_START=100\nRANDOMNESS=0\nLAP_GAIN=0\n'
