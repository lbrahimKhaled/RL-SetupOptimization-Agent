"""SAC policy with LayerNorm critics (D2 spec §3).

`LayerNormContinuousCritic`: SB3's `ContinuousCritic` whose Q-networks have a LayerNorm after
every hidden Linear (before the activation), none after the output Linear (SB3 `create_mlp`
adds post_linear_modules only to hidden layers). The actor is SB3's unchanged `Actor`.
`LNSACPolicy.make_critic` builds this class for the critic and the critic target alike.
"""
from torch import nn
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy


class LayerNormContinuousCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim,
                 activation_fn=nn.ReLU, normalize_images=True, n_critics=2, share_features_extractor=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim,
                         activation_fn=activation_fn, normalize_images=normalize_images,
                         n_critics=n_critics, share_features_extractor=share_features_extractor)
        action_dim = get_action_dim(self.action_space)
        for idx in range(n_critics):   # replace SB3's plain MLPs (same names -> same state_dict keys)
            q = nn.Sequential(*create_mlp(features_dim + action_dim, 1, net_arch, activation_fn,
                                          post_linear_modules=[nn.LayerNorm]))
            self.add_module(f'qf{idx}', q)
            self.q_networks[idx] = q


class LNSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
