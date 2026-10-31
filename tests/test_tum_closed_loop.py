"""Pure helpers of scripts/comparison/tum_closed_loop.py (sign conventions, limiter, envelope,
segmenting, classification); no OCD physics (the debug worktree is not needed)."""
from pathlib import Path
import math
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/comparison'))
import tum_closed_loop as cl  # noqa: E402


def test_lateral_command_signs_and_feedforward():
    g = cl.Gains()
    steer0, k0 = cl.lateral_command(0., 0., 30., .01, g, 3.6)
    assert k0 == .01
    assert steer0 == pytest.approx(math.atan(.036) + g.k_us_rad_per_mps2 * 900 * .01)
    # left of the line (e > 0) or pointing left (chi > 0): steer right relative to the feed-forward
    assert cl.lateral_command(1., 0., 30., .01, g, 3.6)[0] < steer0
    assert cl.lateral_command(0., .05, 30., .01, g, 3.6)[0] < steer0
    # spatial bandwidth falls with speed (gentler at 80 m/s than at 20 m/s for the same offset)
    d_slow = steer0 - cl.lateral_command(1., 0., 20., .01, g, 3.6)[0]
    d_fast = cl.lateral_command(0., 0., 80., .01, g, 3.6)[0] - cl.lateral_command(1., 0., 80., .01, g, 3.6)[0]
    assert d_slow > 0 and d_fast > 0


def test_slip_limiter_cuts_and_recovers():
    f = 1.
    f = cl.slip_limiter(f, .2, .08, .6, .1)
    assert f == pytest.approx(.6)
    f = cl.slip_limiter(f, -.2, .08, .6, .1)          # |slip| (ABS passes negative slip)
    assert f == pytest.approx(.36)
    for _ in range(10):
        f = cl.slip_limiter(f, .0, .08, .6, .1)
    assert f == 1.


def test_envelope_exponent():
    assert cl.envelope_exponent(1.) == pytest.approx(2.)
    assert cl.envelope_exponent(math.sqrt(2.) / 2.) == pytest.approx(1.)
    assert cl.envelope_exponent(.95) < 2.


def _mf_params():
    keys = dict(FNOMIN=3000., LFZO=1., LHX=1., LVX=1., LMUX=.97, LCX=1., LEX=1., LKX=1., LKY=1.,
                LHY=1., LCY=1., LMUY=.97, LEY=1., LVY=1., LXAL=1., LYKA=1., LVYKA=1.,
                PCX1=1.6, PDX1=1.70103, PDX2=-.15464, PEX1=.7, PEX2=-.17, PEX3=.023, PEX4=-.14,
                PKX1=30.7, PKX2=.27, PKX3=.13, PHX1=0., PHX2=0., PVX1=0., PVX2=0.,
                PCY1=1.5, PDY1=1.90722, PDY2=-.15464, PEY1=.1, PEY2=-1.2, PEY3=.05,
                PKY1=-75.5, PKY2=4.65, PKY4=2., PHY1=.003, PHY2=-.0006, PVY1=.04, PVY2=-.02,
                RBX1=17.4, RBX2=12.9, RCX1=1.1, REX1=0., REX2=0., RHX1=.001,
                RBY1=20.6, RBY2=-23.5, RBY3=.001, RCY1=1., REY1=0., REY2=0., RHY1=-.02, RHY2=0.,
                RVY1=.16, RVY2=.03, RVY4=-29.7, RVY5=.03, RVY6=0.)
    return keys.__getitem__


def test_mf52_pure_peak_matches_tum_law_and_combined_reduces():
    p = _mf_params()
    kap = np.linspace(-.6, .6, 2401)
    fx, _ = cl.mf52_forces(p, 3000., kap, 0. * kap)
    assert fx.max() / 3000. == pytest.approx(1.65, rel=1e-3)        # TUM front mux at fz_0
    fx_c, _ = cl.mf52_forces(p, 3000., .1, .1)
    fx_p, _ = cl.mf52_forces(p, 3000., .1, 0.)
    assert abs(fx_c) < abs(fx_p)                                      # slip angle reduces Fx


class _Prof:
    def __init__(self):
        self.L = 1000.
        self.s = np.arange(0., 1000., 5.)
        self.v = 50. + 20. * np.cos(2 * np.pi * self.s / 500.)        # two braking points
        self.t = np.r_[0., np.cumsum(5. / self.v[:-1])]
        self.ax = np.zeros_like(self.s)
        self.tum = {'ay_mps2': np.zeros_like(self.s)}

    def v_at(self, s):
        return np.interp(np.asarray(s) % self.L, self.s, self.v, period=self.L)


def test_corner_segments_cut_at_speed_maxima():
    cuts = cl.corner_segments(_Prof())
    assert cuts == [0., 500.]


def test_classify_priority():
    base = dict(brake=0., throttle=.5, throttle_ff=.5, tc_factor=1., mu_use_front=.5, mu_use_rear=.5)
    assert cl.classify(base) == 'controller'
    assert cl.classify({**base, 'mu_use_front': .95}) == 'lateral_grip'
    assert cl.classify({**base, 'throttle': 1.}) == 'power'
    assert cl.classify({**base, 'tc_factor': .5}) == 'traction'
    assert cl.classify({**base, 'brake': .5, 'tc_factor': .5}) == 'braking'
