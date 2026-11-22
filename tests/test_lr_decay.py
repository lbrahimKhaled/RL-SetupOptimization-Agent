"""TrainConfig.lr_decay (Ibrahim, 18 Nov 2026; DECISIONS #30): linear decay of the actor and critic lrs to
floor x base between start_step and end_step, alpha untouched; off = bit-identical lrs; resume-overridable."""
import json
from types import SimpleNamespace
import pytest
from rlsetup.driving.callbacks import LrDecayCallback
from rlsetup.driving.train import RESUME_OVERRIDABLE
from rlsetup.driving.train_config import (PRESETS, LrDecayConfig, TrainConfig, apply_overrides, config_json,
                                          train_config_from_dict)
from sb3_helpers import make_model


def test_scale_is_linear_from_start_to_end_then_holds_the_floor():
    c = LrDecayConfig(enabled=True, start_step=100, end_step=300, floor=.1)
    assert [c.scale(s) for s in (0, 100, 200, 300, 10**9)] == pytest.approx([1., 1., .55, .1, .1])
    assert LrDecayConfig().scale(500) is None


@pytest.mark.parametrize('kw', [dict(start_step=300, end_step=300), dict(start_step=-1, end_step=10),
                                dict(end_step=10, floor=0.), dict(end_step=10, floor=1.5)])
def test_invalid_schedules_are_refused_at_run_start(kw):
    bad = LrDecayConfig(enabled=True, **kw)     # constructible: --set overrides arrive one field at a time
    for use in (bad.check, lambda: bad.scale(0), lambda: LrDecayCallback(bad)):
        with pytest.raises(ValueError):
            use()
    LrDecayConfig(enabled=False, **kw).check()  # only checked when on


def test_off_is_omitted_from_config_json_and_on_round_trips():
    for name, make in PRESETS.items():
        assert ('"lr_decay"' in config_json(make())) == (name in ('bcn_f1_r4_retry', 'bcn_f1_r5', 'bcn_f1_r6', 'bcn_f1_r6_smoke', 'bcn_f1_r6_scratch', 'bcn_f1_r6_scratch_smoke')), name   # the only ones with it on
    c = apply_overrides(PRESETS['bcn_f1_r4'](), ['lr_decay.enabled=True', 'lr_decay.start_step=7800000',
                                                 'lr_decay.end_step=9800000', 'lr_decay.floor=0.1'])
    j = json.loads(config_json(c))
    assert j['lr_decay'] == {'enabled': True, 'start_step': 7_800_000, 'end_step': 9_800_000, 'floor': .1}
    assert train_config_from_dict(j) == c
    assert train_config_from_dict(json.loads(config_json(TrainConfig()))).lr_decay == LrDecayConfig()


def test_resume_may_switch_it_on():
    assert {'lr_decay.enabled', 'lr_decay.start_step', 'lr_decay.end_step', 'lr_decay.floor'} <= RESUME_OVERRIDABLE


def _lrs(model):
    model._update_learning_rate([model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer])
    return [o.param_groups[0]['lr'] for o in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer)]


def test_scale_multiplies_actor_and_critic_lrs_but_not_alpha():
    model = make_model(fill=0)
    base = model.lr_schedule(1.)
    assert model.lr_scale is None and _lrs(model) == pytest.approx([base, base, base])      # off: SB3 as before
    model.actor_lr = 1e-4
    assert _lrs(model) == pytest.approx([1e-4, base, base])
    model.lr_scale = .5
    assert _lrs(model) == pytest.approx([5e-5, base * .5, base])
    model.actor_lr = None
    assert _lrs(model) == pytest.approx([base * .5, base * .5, base])
    del model.lr_scale                                      # a model pickled before lr_scale existed
    assert _lrs(model) == pytest.approx([base, base, base])


def test_lr_scale_survives_save_and_load(tmp_path):
    model = make_model(fill=0)
    model.lr_scale = .25
    model.save(tmp_path / 'm.zip')
    from rlsetup.driving.agent import StagedSAC
    assert StagedSAC.load(tmp_path / 'm.zip', device='cpu').lr_scale == .25


def test_callback_sets_the_scale_from_the_step_and_clears_it_when_off():
    model = SimpleNamespace(num_timesteps=200, lr_scale=.3, get_env=lambda: None, logger=None)
    cb = LrDecayCallback(LrDecayConfig(enabled=True, start_step=100, end_step=300, floor=.1))
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert model.lr_scale == pytest.approx(.55)
    model.num_timesteps = 400
    cb.on_step()
    assert model.lr_scale == pytest.approx(.1)
    off = LrDecayCallback(LrDecayConfig())
    off.init_callback(model)
    off.on_step()
    assert model.lr_scale is None
