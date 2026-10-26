"""R1 Amendment 1 §C1, §C2: the relabelled ref_tracking buffer against time_pbrs ground truth."""
import dataclasses
import numpy as np
import pytest
from rlsetup.driving.config import EnvConfig, RewardConfig
from rlsetup.driving.relabel import relabel_to_time_pbrs
from relabel_truth import collect

TOL_OK, TOL_FAIL = 1e-3, .05          # spec C1: non-failure rows / failure rows


@pytest.fixture(scope='module')
def truth(make_env, track):
    leg, r1, names, cfgs = collect(make_env, track, make_env().ref)
    n = leg.pos
    r_old = leg.rewards[:n].copy()
    stats = relabel_to_time_pbrs(leg, *cfgs)
    return leg, r1, names, n, r_old, stats


def test_relabel_matches_env_ground_truth(truth):
    """C1: same seeds / actions under both rewards; flags equal exactly, rewards within tolerance.
    Max errors measured (26 Oct): non-failure 9.0e-5, failure 6.1e-5."""
    leg, r1, names, n, r_old, stats = truth
    kinds = {nm.split('_')[0] for nm in names}
    assert {'off', 'stall', 'time', 'wrong'} <= kinds
    np.testing.assert_array_equal(leg.dones[:n], r1.dones[:n])
    np.testing.assert_array_equal(leg.timeouts[:n], r1.timeouts[:n])
    fail = (r1.dones[:n, 0] > .5) & ~(r1.timeouts[:n, 0] > .5)
    stall = (r1.dones[:n, 0] > .5) & (r1.timeouts[:n, 0] > .5) & np.array([nm.startswith('stall') for nm in names])
    err = np.abs(leg.rewards[:n, 0].astype(float) - r1.rewards[:n, 0].astype(float))
    old_fail = fail | stall                                       # failures under the old reward
    print(f'relabel max error: non-failure {err[~old_fail].max():.2e}, failure {err[old_fail].max():.2e}')
    assert err[~old_fail].max() <= TOL_OK and err[old_fail].max() <= TOL_FAIL
    assert stats['failures_old'] == int(old_fail.sum()) and stats['stalls'] == int(stall.sum()) == 3
    assert stats['terminations_new'] == int(fail.sum()) and stats['nonfinite'] == 0
    assert np.abs(r_old[:n, 0] - r1.rewards[:n, 0]).max() > 1.                  # the relabel did something


def test_relabelled_nstep_samples_match(truth):
    """C2: SB3 n-step samples (returns, discounts, final dones, next obs) for the same indices."""
    leg, r1, names, n, *_ = truth
    inds = np.arange(n - 1)
    out = []
    for rb in (leg, r1):
        np.random.seed(0)
        out.append(rb._get_samples(inds))
    a, b = out
    np.testing.assert_array_equal(a.dones.numpy(), b.dones.numpy())
    np.testing.assert_array_equal(a.discounts.numpy(), b.discounts.numpy())
    np.testing.assert_array_equal(a.next_observations.numpy(), b.next_observations.numpy())
    err = np.abs(a.rewards.numpy() - b.rewards.numpy())[:, 0]
    assert err.max() <= TOL_FAIL and np.median(err) <= TOL_OK


def test_relabel_refuses_wrong_configs(truth):
    leg, *_ = truth
    src, dst = EnvConfig(), EnvConfig(reward=RewardConfig(mode='time_pbrs'))
    for a, b in ((dst, dst), (src, src), (src, dataclasses.replace(dst, reward=RewardConfig(mode='time_pbrs', c=.2))),
                 (src, dataclasses.replace(dst, action_repeat=1))):
        with pytest.raises(ValueError):
            relabel_to_time_pbrs(leg, a, b)
