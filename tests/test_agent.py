"""D2 §10: LayerNorm critic, StagedSAC freezing / actor_lr / save-load, output-bias init, configs."""
import dataclasses
import json
import gymnasium as gym
import numpy as np
import pytest
import torch as th
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv
from rlsetup.driving.agent import StagedSAC, build_model, init_output_bias, SB3_VERSION
from rlsetup.driving.policy import LayerNormContinuousCritic
from rlsetup.driving.train_config import (TrainConfig, PRESETS, apply_overrides, train_config_from_dict,
                                          config_json)

OBS = 159


class TinyEnv(gym.Env):
    """Numpy-only stand-in with the driving env's spaces."""
    observation_space = gym.spaces.Box(-10., 10., (OBS,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.np_random.uniform(-1, 1, OBS).astype(np.float32), {}

    def step(self, action):
        obs = self.np_random.uniform(-1, 1, OBS).astype(np.float32)
        return obs, float(self.np_random.normal()), False, False, {}


def tiny_cfg(**kw):
    c = TrainConfig()
    return dataclasses.replace(c, n_envs=1, agent=dataclasses.replace(c.agent, buffer_size=5_000, **kw))


def make_model(cfg=None, fill=2_000):
    cfg = cfg or tiny_cfg()
    model = build_model(cfg, DummyVecEnv([TinyEnv]))
    rng = np.random.default_rng(0)
    rb = model.replay_buffer
    for _ in range(fill):
        rb.add(rng.uniform(-1, 1, (1, OBS)).astype(np.float32), rng.uniform(-1, 1, (1, OBS)).astype(np.float32),
               rng.uniform(-1, 1, (1, 3)).astype(np.float32), rng.normal(size=1).astype(np.float32),
               rng.uniform(size=1) < .01, [{}])
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(None, []))
    return model


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same(a, b):
    return all(th.equal(a[k], b[k]) for k in a)


def all_differ(a, b):
    """Every weight/bias tensor changed."""
    return all(not th.equal(a[k], b[k]) for k in a)


def test_sb3_version_pinned():
    import stable_baselines3, sb3_contrib
    assert stable_baselines3.__version__ == SB3_VERSION == '2.9.0'
    assert sb3_contrib.__version__ == '2.9.0'


def test_layernorm_critic_structure():
    model = make_model(fill=0)
    for critic in (model.critic, model.critic_target):
        assert type(critic) is LayerNormContinuousCritic
        assert len(critic.q_networks) == 2
        for q in critic.q_networks:
            mods = list(q)
            ln = [i for i, m in enumerate(mods) if isinstance(m, nn.LayerNorm)]
            assert len(ln) == 3
            assert all(isinstance(mods[i - 1], nn.Linear) and isinstance(mods[i + 1], nn.ReLU) for i in ln)
            assert isinstance(mods[-1], nn.Linear) and mods[-1].out_features == 1   # no LN on the output
            assert [m.out_features for m in mods if isinstance(m, nn.Linear)] == [256, 256, 256, 1]
        # the q_networks list is what forward() uses: it must hold the registered LN modules
        assert all(q is getattr(critic, f'qf{i}') for i, q in enumerate(critic.q_networks))
    assert not any(isinstance(m, nn.LayerNorm) for m in model.actor.modules())
    assert model.policy.share_features_extractor is False
    assert model.gamma == .992 and model.n_steps == 3 and model.batch_size == 512
    assert model.learning_starts == 0 and model.target_entropy == -3. and model.alpha == pytest.approx(.1)
    assert type(model.replay_buffer).__name__ == 'NStepReplayBuffer'


def test_frozen_train_keeps_actor_and_alpha_bit_identical():
    model = make_model()
    model.set_frozen(actor=True, ent_coef=True)
    actor0, critic0, target0 = snapshot(model.actor), snapshot(model.critic), snapshot(model.critic_target)
    alpha0 = model.log_ent_coef.detach().clone()
    model.train(5, 64)
    assert same(actor0, snapshot(model.actor))
    assert th.equal(alpha0, model.log_ent_coef.detach())
    assert model.ent_coef_optimizer is not None                  # restored after train()
    assert all_differ(critic0, snapshot(model.critic))
    assert all_differ(target0, snapshot(model.critic_target))
    assert model.logger.name_to_value['train/ent_coef'] == pytest.approx(.1)


def test_unfrozen_train_changes_everything():
    model = make_model()
    model.set_frozen(actor=True, ent_coef=True)
    model.set_frozen(actor=False, ent_coef=False)
    actor0, critic0 = snapshot(model.actor), snapshot(model.critic)
    alpha0 = model.log_ent_coef.detach().clone()
    model.train(5, 64)
    assert all_differ(actor0, snapshot(model.actor))
    assert all_differ(critic0, snapshot(model.critic))
    assert not th.equal(alpha0, model.log_ent_coef.detach())


def test_actor_lr_override():
    model = make_model()
    model.actor_lr = 1e-4
    model.train(1, 64)
    assert [g['lr'] for g in model.actor.optimizer.param_groups] == [1e-4]
    assert [g['lr'] for g in model.critic.optimizer.param_groups] == [3e-4]
    assert [g['lr'] for g in model.ent_coef_optimizer.param_groups] == [3e-4]
    assert model.logger.name_to_value['train/actor_lr'] == 1e-4


def test_save_load_round_trip(tmp_path):
    model = make_model()
    init_output_bias(model.actor, pedal_deadzone=True)
    model.train(3, 64)
    model.set_frozen(actor=True, ent_coef=True)
    model.actor_lr = 1e-4
    model.save(tmp_path / 'm.zip')
    loaded = StagedSAC.load(tmp_path / 'm.zip', env=DummyVecEnv([TinyEnv]), device='cpu')
    obs = np.random.default_rng(1).uniform(-1, 1, (16, OBS)).astype(np.float32)
    a0, _ = model.predict(obs, deterministic=True)
    a1, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_array_equal(a0, a1)
    assert type(loaded.critic) is LayerNormContinuousCritic
    assert loaded.freeze_actor and loaded.freeze_ent_coef and loaded.actor_lr == 1e-4
    assert not any(p.requires_grad for p in loaded.actor.parameters())   # re-applied after load
    assert th.equal(loaded.log_ent_coef, model.log_ent_coef)
    assert loaded.ent_coef_optimizer is not None


def test_output_bias_init():
    model = make_model(fill=0)
    init_output_bias(model.actor, pedal_deadzone=True)
    obs = th.zeros(1, OBS)
    with th.no_grad():
        a = model.actor(obs, deterministic=True).numpy()[0]
        _, log_std, _ = model.actor.get_action_dist_params(obs)
    np.testing.assert_allclose(a, [0., -.275, -.8], atol=.01)
    np.testing.assert_allclose(log_std.numpy()[0], -1., atol=.01)
    np.testing.assert_allclose(model.actor.mu.bias.detach().numpy(), [0., -.2822, -1.0986], atol=1e-4)
    with pytest.raises(AssertionError):
        init_output_bias(model.actor, pedal_deadzone=False)


def test_presets_and_overrides():
    full, smoke = PRESETS['full'](), PRESETS['smoke']()
    assert (full.n_envs, full.torch_threads, full.gradient_steps, full.utd) == (4, 1, 2, .5)
    assert smoke.preset == 'smoke' and smoke.max_wall_h == pytest.approx(1 / 3)
    assert smoke.warmup.offline_blocks * smoke.warmup.offline_steps_per_block == 5_000
    assert smoke.warmup.online_warmup_steps == 10_000 and smoke.bc.max_epochs == 10
    assert smoke.guard.interval == 5_000 and smoke.eval.interval == 25_000 and smoke.eval.stochastic_laps == 3
    c = apply_overrides(full, ['agent.batch_size=256', 'env.reward.c=0.05', 'max_wall_h=None',
                               'demos.noise_std=[0.05, 0.1, 0.1]', 'runs_root=/tmp/x'])
    assert c.agent.batch_size == 256 and c.env.reward.c == .05 and c.max_wall_h is None
    assert c.demos.noise_std == (.05, .1, .1) and c.runs_root == '/tmp/x'
    with pytest.raises(KeyError):
        apply_overrides(full, ['agent.nope=1'])
    back = train_config_from_dict(json.loads(config_json(c)))
    assert back == c
