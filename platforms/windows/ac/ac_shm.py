"""Read-only view of Assetto Corsa's shared memory (acpmf_physics / acpmf_graphics), Windows only.

Used by ac_supervisor.py to tell whether AC is on track and which setup is loaded: values the setup
fixes (fuel, brake bias, engine brake, ERS levels) and the track grip are compared with a fingerprint taken from a known-good
session. python platforms/windows/ac/ac_shm.py prints the current values.
"""
import ctypes as ct
import json
import mmap

F, I = ct.c_float, ct.c_int32


class Physics(ct.Structure):
    _pack_ = 4
    _fields_ = [('packetId', I), ('gas', F), ('brake', F), ('fuel', F), ('gear', I), ('rpms', I),
                ('steerAngle', F), ('speedKmh', F), ('velocity', F*3), ('accG', F*3), ('wheelSlip', F*4),
                ('wheelLoad', F*4), ('wheelsPressure', F*4), ('wheelAngularSpeed', F*4), ('tyreWear', F*4),
                ('tyreDirtyLevel', F*4), ('tyreCoreTemperature', F*4), ('camberRAD', F*4),
                ('suspensionTravel', F*4), ('drs', F), ('tc', F), ('heading', F), ('pitch', F), ('roll', F),
                ('cgHeight', F), ('carDamage', F*5), ('numberOfTyresOut', I), ('pitLimiterOn', I), ('abs', F),
                ('kersCharge', F), ('kersInput', F), ('autoShifterOn', I), ('rideHeight', F*2), ('turboBoost', F),
                ('ballast', F), ('airDensity', F), ('airTemp', F), ('roadTemp', F), ('localAngularVel', F*3),
                ('finalFF', F), ('performanceMeter', F), ('engineBrake', I), ('ersRecoveryLevel', I),
                ('ersPowerLevel', I), ('ersHeatCharging', I), ('ersIsCharging', I), ('kersCurrentKJ', F),
                ('drsAvailable', I), ('drsEnabled', I), ('brakeTemp', F*4), ('clutch', F), ('tyreTempI', F*4),
                ('tyreTempM', F*4), ('tyreTempO', F*4), ('isAIControlled', I), ('tyreContactPoint', F*12),
                ('tyreContactNormal', F*12), ('tyreContactHeading', F*12), ('brakeBias', F)]


class Graphics(ct.Structure):
    _pack_ = 4
    _fields_ = [('packetId', I), ('status', I), ('session', I), ('currentTime', ct.c_wchar*15),
                ('lastTime', ct.c_wchar*15), ('bestTime', ct.c_wchar*15), ('split', ct.c_wchar*15),
                ('completedLaps', I), ('position', I), ('iCurrentTime', I), ('iLastTime', I), ('iBestTime', I),
                ('sessionTimeLeft', F), ('distanceTraveled', F), ('isInPit', I), ('currentSectorIndex', I),
                ('lastSectorTime', I), ('numberOfLaps', I), ('tyreCompound', ct.c_wchar*33),
                ('replayTimeMultiplier', F), ('normalizedCarPosition', F), ('carCoordinates', F*3),
                ('penaltyTime', F), ('flag', I), ('idealLineOn', I), ('isInPitLane', I), ('surfaceGrip', F),
                ('mandatoryPitDone', I), ('windSpeed', F), ('windDirection', F)]


def _read(tag, shape):
    m = mmap.mmap(-1, ct.sizeof(shape), tagname='acpmf_'+tag, access=mmap.ACCESS_READ)
    try:
        return shape.from_buffer_copy(m[:ct.sizeof(shape)])
    finally:
        m.close()


def snapshot():
    """Current values, or None if AC's shared memory is not there."""
    try:
        p, g = _read('physics', Physics), _read('graphics', Graphics)
    except (OSError, ValueError):
        return None
    return dict(physics_packet=p.packetId, graphics_packet=g.packetId, status=g.status, in_pit=g.isInPit,
                surface_grip=round(g.surfaceGrip, 4), wind_speed=round(g.windSpeed, 2),
                speed_kmh=round(p.speedKmh, 2), gear=p.gear, gas=round(p.gas, 3), brake=round(p.brake, 3),
                setup=dict(fuel=round(p.fuel, 2), brake_bias=round(p.brakeBias, 4), engine_brake=p.engineBrake,
                           ers_recovery=p.ersRecoveryLevel, ers_power=p.ersPowerLevel,
                           ers_heat_charging=p.ersHeatCharging, ballast=round(p.ballast, 1)))


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=1))
