"""Demo / settle controller (review R5) and the pedal <-> action mapping (review R3).

`DemoDriver` = pure pursuit on xi_ref with `ocd_lap.driver.ScriptedDriver` geometry (rear-axle
reference, lookahead 3 m + 0.35 s * vx, steering atan(2 L sin(alpha) / d)) plus a speed PI on
the pedals with the gains of `ocd_lap.manoeuvres._speed_pedals` (0.4 per m/s, 0.2 per m),
the brake pedal capped at 0.3 (3.6 kN m at 12 kN m total, below the 3.65 kN m no-aero lock
limit), and (beyond R5, DemoConfig.traction_control) ScriptedDriver's rear-slip traction control
on the throttle. It is not `ScriptedDriver`'s PI, which caps drive torque at ~270 N m.
It outputs ENV ACTIONS ([-1, 1]^3, the inverse of the env's pedal mapping), so the same object
drives the settle phase and, through `DemoPolicy`, produces D2 demonstration data.

Pedal mapping (R3), throttle and brake alike:
    pedal_deadzone=True:  pedal = clip((a + 0.5) / 1.5, 0, 1); inverse a = -0.5 + 1.5 pedal for
                          pedal > 0, a = released_pedal_action (-0.8) for pedal = 0
    pedal_deadzone=False: pedal = (a + 1) / 2 (ocd_first_lap OCDRacingEnv); inverse a = 2 pedal - 1
Steering: steering_rad = a * 0.6 rad (MAX_STEER_RAD), positive = left (OCD yaw convention).
"""
import math
import numpy as np
from ocd_lap.backend import Inputs
from ocd_lap import cars
from ocd_lap.env import MAX_STEER_RAD


def action_to_inputs(a, pedal_deadzone):
    """Validated action array (3,) -> pedal `Inputs`."""
    if pedal_deadzone:
        pedal = lambda x: float(min(1., max(0., (x + .5) / 1.5)))
        return Inputs(float(a[0]) * MAX_STEER_RAD, pedal(a[1]), pedal(a[2]))
    # identical expression to OCDRacingEnv.to_inputs (bitwise-equal inputs)
    return Inputs(float(a[0]) * MAX_STEER_RAD, (a[1] + 1) / 2, (a[2] + 1) / 2)


def pedal_to_action(pedal, pedal_deadzone, released_action=-.8):
    if pedal_deadzone:
        return -.5 + 1.5 * pedal if pedal > 0. else released_action
    return 2. * pedal - 1.


def inputs_to_action(steering_rad, throttle, brake, pedal_deadzone, released_action=-.8):
    """Pedals -> env action (float32), the inverse of `action_to_inputs`."""
    return np.array([float(np.clip(steering_rad / MAX_STEER_RAD, -1., 1.)),
                     pedal_to_action(throttle, pedal_deadzone, released_action),
                     pedal_to_action(brake, pedal_deadzone, released_action)], dtype=np.float32)


class DemoDriver:
    """Pure pursuit on `ref` + speed PI. `control_dt_s` is the interval between `act` calls."""

    def __init__(self, ref, cfg, *, pedal_deadzone, control_dt_s, car='av21'):
        self.ref, self.cfg = ref, cfg
        spec = cars.get(car)
        self.rear_axle_m, self.wheelbase_m = spec.rear_axle_m, spec.wheelbase_m
        self.pedal_deadzone = pedal_deadzone
        self.dt = control_dt_s
        self.integral = 0.

    def reset(self, integral=0.):
        self.integral = float(integral)

    def steering(self, state, s_ref):
        """Pure-pursuit steering angle [rad] (ScriptedDriver.act geometry, goal on xi_ref)."""
        yaw = state['yaw']
        rear = np.array([state['x'], state['y']]) - self.rear_axle_m * np.array([math.cos(yaw), math.sin(yaw)])
        lookahead = self.cfg.lookahead_m + self.cfg.lookahead_s * max(state['vx'], 0.)
        d = self.ref.point(s_ref - self.rear_axle_m + lookahead) - rear
        alpha = math.atan2(d[1], d[0]) - yaw
        return float(np.clip(math.atan2(2. * self.wheelbase_m * math.sin(alpha), np.linalg.norm(d)),
                             -MAX_STEER_RAD, MAX_STEER_RAD))

    def pedals(self, target_mps, vx, rear_slip=0.):
        """Speed PI -> (throttle, brake) in [0, 1]; brake capped at cfg.brake_cap; throttle
        scaled down by the traction control when |rear slip ratio| > traction_slip_start."""
        c = self.cfg
        error = target_mps - vx
        self.integral = float(np.clip(self.integral + error * self.dt, -c.integral_limit_m, c.integral_limit_m))
        u = c.speed_kp * error + c.speed_ki * self.integral
        throttle = min(1., max(0., u))
        if c.traction_control and rear_slip > c.traction_slip_start:
            throttle *= max(0., 1. - (rear_slip - c.traction_slip_start) / c.traction_slip_span)
        return throttle, min(c.brake_cap, max(0., -u))

    def act(self, state, s_ref, target_mps):
        """Env action for `state` at xi_ref arc length `s_ref`, holding speed `target_mps`."""
        steer = self.steering(state, s_ref)
        throttle, brake = self.pedals(target_mps, state['vx'],
                                      max(abs(state['slip_rl']), abs(state['slip_rr'])))
        return inputs_to_action(steer, throttle, brake, self.pedal_deadzone, self.cfg.released_pedal_action)


class DemoPolicy:
    """Demo driver as a policy for `DrivingEnv` (D2 demonstration data): target speed
    f * v_qss(s + max(vx, 0) * speed_preview_s) on the p0 QSS profile. Like
    `ocd_lap.env.ScriptedPolicy` it reads the env's state and line positions directly, which a
    learned policy cannot. `reset()` after `env.reset()` continues the settle controller's
    integral, so the hand-over is smooth."""

    def __init__(self, env, fraction):
        self.env = env.unwrapped
        self.fraction = float(fraction)
        e = self.env
        self.driver = DemoDriver(e.drive_line, e.cfg.demo, pedal_deadzone=e.cfg.pedal_deadzone,
                                 control_dt_s=e.decision_dt, car=e.cfg.car)

    def reset(self):
        self.driver.reset(self.env.settle_driver.integral)

    def target_speed(self):
        e = self.env
        return self.fraction * float(e.qss.speed(e.s + max(e.state['vx'], 0.) * e.cfg.demo.speed_preview_s))

    def __call__(self, observation=None):
        e = self.env
        return self.driver.act(e.state, e.s_drive, self.target_speed())
