"""Named car setups for the driving env (`EnvConfig.setup`).

  p0      `Setup()`: ocd_lap's knob defaults (the AV-21 car shipped with OCD, converted to knob
          units). Every run before p1_TUM used it.
  p1_TUM  p0 with the knobs that TUM's laptime-simulation fixes in its model (laptimesim/src/
          car.py @ afa0aa7, F1_Shanghai.ini) set to TUM's values. TUM's car has no setup; these
          are the assumptions its point mass makes:
            roll_stiffness_front_share 0.4533  TUM splits lateral load transfer by static axle
                                               load, lr / (lf + lr) = 1.632 / 3.6
            aero_front_share           0.4508  TUM's constant CzA 2.20 front / 2.68 rear
            brake_bias_front           0.49    TUM brakes every wheel at its limit; median
                                               front share of the tyre potential over the
                                               braking points of TUM's `none` lap
                                               (comparison/tum/tum_none.csv, ax < -5 m/s^2)
            lsd_lock_drive             1.0     TUM sums both rear wheels' potential when
                                               accelerating ("a limited slip differential
                                               overcomes this problem", car.calc_f_x_pot)
          Unchanged from p0: springs, dampers, heave, packers, total roll stiffness, ride
          heights, aero level, final drive, camber / toe (TUM has none of these). Pair it with
          EnvConfig.max_brake_torque_Nm = 15 kN m (preset p1_TUM): TUM's tyres take up to
          13.7 kN m total at 79 m/s, and the front axle needs about 14.5 kN m at bias 0.49.
  bcn_best  QSS-best setup for Barcelona (Catalunya), sf70h_tum, ERS/DRS off, 15 kN m brakes
          (preset bcn). scripts/bcn_setup_sweep.py, 39 QSS evaluations, docs/results/bcn/
          SETUP_SELECTION.md. QSS lap on the safe line [s] (12 / 15 kN m): p0 89.233 / 89.031,
          p1_TUM 87.899 / 87.884; one knob at a time around p1_TUM @ 15 kN m, then greedy
          combination: + aero_level 1.0 -> 85.171, + aero_front_share 0.5384 -> 83.310,
          + brake_bias_front 0.55 -> 82.888 (bcn_best; 83.081 at 12 kN m). Not taken:
          roll_stiffness_front_share 0.35 (bound, -0.023 s), lower ride height / rake (the car is
          already below the ground at top speed, no floor contact in OCD). Flags: aero_level at
          its upper bound; understeer gradient -0.02 / -0.19 deg/g at 75 / 90 m/s (p1_TUM +0.12 /
          +0.44), i.e. mildly oversteer at top speed; front ride height at v_max -6.2 cm (p1_TUM -2.8).
"""
from ocd_lap.car_setup import Setup

SETUPS = {
    'p0': Setup(),
    'p1_TUM': Setup().replace(roll_stiffness_front_share=.4533, aero_front_share=.4508,
                              brake_bias_front=.49, lsd_lock_drive=1.),
    'bcn_best': Setup().replace(roll_stiffness_front_share=.4533, aero_front_share=.5384,
                                brake_bias_front=.55, lsd_lock_drive=1., aero_level=1.),
}


def get(name):
    """The Setup called `name` (a key of SETUPS)."""
    if name not in SETUPS:
        raise KeyError(f'unknown setup {name!r} (known: {sorted(SETUPS)})')
    return SETUPS[name]
