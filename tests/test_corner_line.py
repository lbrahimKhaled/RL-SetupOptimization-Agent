"""Corner line penalty (RewardConfig.line_k; Ibrahim, 17 Nov 2026) and preset bcn_f1_r4: per base tick ending
inside a corner window, min(line_k max(0, |e| - line_free_m), line_cap) max(0, ds) leaves the reward; off =
bit-identical; config keys omitted while off; bcn_f1_r4 = f1_r4_scratch on Catalunya + the penalty."""
import dataclasses
import json
from types import SimpleNamespace
import numpy as np
import pytest
from ocd_lap import cars
from rlsetup.driving.config import EnvConfig, RewardConfig, CornerBonusConfig
from rlsetup.driving.env import DrivingEnv
from rlsetup.driving.train_config import PRESETS, config_json, train_config_from_dict, BCN_LINE_K, BCN_W4_LINE_K
import speed_truth as ST

needs_f1 = pytest.mark.skipif('sf70h_f1' not in cars.CARS, reason='sf70h_f1 needs ocd_first_lap branch sf70h-f1')
E3 = CornerBonusConfig(enabled=True, mode='entry_exit')


def test_config_validation():
    RewardConfig(mode='time_pbrs', line_k=.07)
    with pytest.raises(ValueError, match='time_pbrs'):
        RewardConfig(line_k=.07)                                  # ref_tracking
    with pytest.raises(ValueError, match='below a'):
        RewardConfig(mode='time_pbrs', line_k=.07, line_cap=1.)   # cap >= a: forward could lose reward
    for bad in ({'line_k': -1.}, {'line_free_m': -.1}, {'line_cap': float('nan')}):
        with pytest.raises(ValueError):
            RewardConfig(mode='time_pbrs', **bad)


def test_to_dict_omits_the_keys_while_off_and_round_trips():
    assert not {'line_k', 'line_free_m', 'line_cap'} & set(EnvConfig().to_dict()['reward'])
    on = EnvConfig(reward=RewardConfig(mode='time_pbrs', line_k=.07)).to_dict()['reward']
    assert (on['line_k'], on['line_free_m'], on['line_cap']) == (.07, .5, .5)
    for name, make in PRESETS.items():
        cfg = make()
        assert train_config_from_dict(json.loads(config_json(cfg))) == cfg, name
        if not name.startswith(('bcn_f1_r4', 'bcn_f1_r5')):
            assert '"line_k"' not in config_json(cfg), name


def _fake(units, L=1000., k=.1, free=.5, cap=.5, s=0.):
    from rlsetup.driving.corners import CornerTable
    t = CornerTable(length=L, s=np.arange(0., L, 1.), v=np.full(int(L), 50.),
                    units=[dict(name=f'U{i + 1}', start_m=a, exit_m=b) for i, (a, b) in enumerate(units)], lap_time_s=20.)
    return SimpleNamespace(corners=t, s=s, cfg=SimpleNamespace(reward=RewardConfig(mode='time_pbrs', line_k=k,
                                                                                    line_free_m=free, line_cap=cap)))


def test_penalty_arithmetic_window_band_cap_and_direction():
    pen = DrivingEnv._line_penalty
    inside, outside, wrap = _fake([(100., 300.)], s=200.), _fake([(100., 300.)], s=350.), _fake([(900., 50.)], s=10.)
    assert pen(inside, .5, 2.) == 0.                                 # inside the free band
    assert pen(inside, 1.5, 2.) == pytest.approx(.1 * 1. * 2.)        # k (|e| - free) ds
    assert pen(inside, 20., 2.) == pytest.approx(.5 * 2.)             # capped per metre
    assert pen(inside, 1.5, -1.) == 0. and pen(inside, 1.5, 0.) == 0.  # only forward progress is charged
    assert pen(outside, 5., 2.) == 0.                                 # outside every window
    assert pen(wrap, 1.5, 2.) == pytest.approx(.2)                    # window through s = 0
    assert pen(_fake([(100., 300.)], s=100.), 1.5, 1.) == pytest.approx(.1)   # window edges inclusive
    assert pen(_fake([(100., 300.)], s=300.), 1.5, 1.) == pytest.approx(.1)


def test_off_is_bit_identical_and_on_subtracts_exactly_the_reported_penalty(make_env, track, qss):
    """Same rollouts with and without the penalty: identical observations and terminations; line_k = 0 leaves
    the reward untouched (even with another free band); on: r_on = r_off - line_penalty per decision, a
    penalty only in decisions that pass through a corner window, and some penalty is paid."""
    table = ST.table_for(track, qss)
    base = EnvConfig(max_episode_s=20., settle_s=0., corner_bonus=E3,
                     reward=RewardConfig(mode='time_pbrs', speed_k=20.))
    ref = make_env(base, corners=table)
    zero = make_env(dataclasses.replace(base, reward=dataclasses.replace(base.reward, line_free_m=.3)), corners=table)
    on = make_env(dataclasses.replace(base, reward=dataclasses.replace(base.reward, line_k=.2, line_free_m=0.)),
                  corners=table)
    L, total = float(table.length), 0.
    for name, seed, opts, pol in ST.scenarios():
        _, info = ref.reset(seed=seed, options=opts)
        zero.reset(seed=seed, options=opts)
        on.reset(seed=seed, options=opts)
        s_prev, ep = info['s_track'], 0.
        for k in range(1000):
            a = np.asarray(pol(on, table, k), np.float32)
            o0, r0, te, tr, i0 = ref.step(a)
            oz, rz, *_ = zero.step(a)
            o1, r1, te1, tr1, i1 = on.step(a)
            assert o1.tobytes() == o0.tobytes() and oz.tobytes() == o0.tobytes() and (te1, tr1) == (te, tr)
            assert rz == r0 and 'line_penalty' not in i0
            assert i1['line_penalty'] >= 0. and r1 == pytest.approx(r0 - i1['line_penalty'], abs=1e-9)
            if i1['line_penalty'] > 0.:   # some point of this decision's path lies in a window
                span = (i0['s_track'] - s_prev) % L
                assert any(table.unit_at((s_prev + x) % L) is not None for x in np.linspace(0., span, 50)), name
            ep += i1['line_penalty']
            s_prev = i0['s_track']
            if te or tr:
                break
        assert i1['episode_line_penalty'] == pytest.approx(ep, abs=1e-9)
        total += ep
    assert total > 0.


@needs_f1
def test_preset_bcn_f1_r4_is_f1_r4_scratch_on_catalunya_plus_the_penalty():
    src, c = PRESETS['f1_r4_scratch'](), PRESETS['bcn_f1_r4']()
    assert (c.env.track, c.env.setup, c.env.reference) == ('Catalunya', 'bcn_f1_best',
                                                           'csv:docs/results/bcn_line_setup/line_iqp_w3.csv')
    rw = c.env.reward
    assert (rw.line_k, rw.line_free_m, rw.line_cap) == (BCN_LINE_K, .5, .5) and BCN_LINE_K == .0705
    assert (c.total_timesteps, c.max_wall_h, c.stop.plateau) == (1_000_000_000, None, False)
    env = dataclasses.replace(c.env, track=src.env.track, setup=src.env.setup, reference=src.env.reference,
                              reward=dataclasses.replace(rw, line_k=0.))
    assert dataclasses.replace(c, preset=src.preset, env=env, total_timesteps=src.total_timesteps,
                               max_wall_h=src.max_wall_h, stop=src.stop) == src
    from rlsetup.driving import setups
    s, b = setups.get('bcn_f1_best'), setups.get('sha_f1_best')
    assert s.heave_freq_front_hz == 10. and s.replace(heave_freq_front_hz=b.heave_freq_front_hz) == b


@needs_f1
def test_preset_bcn_f1_r4_w4_is_bcn_f1_r4_on_iqp_w4_with_a_02m_band():
    a, c = PRESETS['bcn_f1_r4'](), PRESETS['bcn_f1_r4_w4']()
    assert c.env.reference == 'csv:docs/results/bcn_line_setup/line_iqp_w4.csv'
    rw = c.env.reward
    assert (rw.line_k, rw.line_free_m, rw.line_cap) == (BCN_W4_LINE_K, .2, .5) and BCN_W4_LINE_K == .0699
    env = dataclasses.replace(c.env, reference=a.env.reference,
                              reward=dataclasses.replace(rw, line_k=BCN_LINE_K, line_free_m=.5))
    assert dataclasses.replace(c, preset=a.preset, env=env) == a
