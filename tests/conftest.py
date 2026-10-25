import pytest
from ocd_lap.track import Track
from rlsetup.driving import qss_profile
from rlsetup.driving.config import EnvConfig
from rlsetup.driving.env import DrivingEnv


@pytest.fixture(scope='session')
def track():
    return Track('Shanghai')   # built once (R1)


@pytest.fixture(scope='session')
def qss(track):
    return qss_profile.load_or_build(track)


@pytest.fixture(scope='session')
def make_env(track, qss):
    def make(config=EnvConfig(), **kw):
        return DrivingEnv(config, track=track, qss=qss, **kw)
    return make


@pytest.fixture(scope='session')
def env(make_env):
    return make_env()
