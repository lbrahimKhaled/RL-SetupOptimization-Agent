"""Behaviour cloning of the SAC actor on the demonstration data (D2 spec §4 step 3).

Loss = -mean log pi(clip(a, +-0.995) | s) of the EXECUTED (noisy) actions under the actor's
squashed Gaussian (`actor.action_dist.proba_distribution(mean, log_std)`), so tanh(mean) fits
the clean demo action and sigma ~ the noise level. Own Adam (not actor.optimizer, which stays
fresh for RL). 90/10 split by episode (BCConfig.split_seed, the same for every training seed),
minibatch order from torch.Generator(seed); early stop on validation NLL with patience; the best
epoch's weights are kept.
"""
import copy
import time
import numpy as np
import torch as th
from .demos import with_index

SLIP_OBS = slice(7, 9)   # DrivingEnv ego block: rear slip ratio RL, RR times ObsConfig.slip_gain


def _nll(actor, obs, act, clip):
    mean, log_std, _ = actor.get_action_dist_params(obs)
    dist = actor.action_dist.proba_distribution(mean, log_std)
    return -dist.log_prob(act.clamp(-clip, clip)).mean()


def _split(n_episodes, cfg):
    if n_episodes < 2:
        raise ValueError('BC needs at least 2 demo episodes (split by episode)')
    perm = np.random.default_rng(cfg.split_seed).permutation(n_episodes)
    n_val = min(max(1, int(round(cfg.val_fraction * n_episodes))), n_episodes - 1)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def pedal(a, pedal_deadzone=True):
    """Action -> pedal in [0, 1] (demo_driver.action_to_inputs)."""
    a = np.asarray(a, dtype=float)
    return np.clip((a + .5) / 1.5, 0., 1.) if pedal_deadzone else (a + 1.) / 2.


def train_bc(actor, demos, cfg, *, seed, slip_gain=5., pedal_deadzone=True, log=print):
    """Fit `actor` (SB3 SAC Actor, modified in place to the best epoch) to `demos`
    (`demos.load_demos` dict). Returns a JSON-ready report."""
    t0 = time.perf_counter()
    if not all(p.requires_grad for p in actor.parameters()):
        raise RuntimeError('actor is frozen')
    d = demos if 'obs_index' in demos else with_index(demos)
    train_eps, val_eps = _split(len(d['ep_start_tr']), cfg)
    in_val = np.isin(d['episode'], val_eps)
    obs = th.as_tensor(d['obs'][d['obs_index']], dtype=th.float32)
    act = th.as_tensor(d['actions'], dtype=th.float32)
    tr_i = th.as_tensor(np.flatnonzero(~in_val))
    va_i = th.as_tensor(np.flatnonzero(in_val))
    gen = th.Generator().manual_seed(int(seed))
    opt = th.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    actor.set_training_mode(True)

    def val_nll():
        with th.no_grad():
            tot = 0.
            for chunk in va_i.split(8192):
                tot += float(_nll(actor, obs[chunk], act[chunk], cfg.action_clip)) * len(chunk)
            return tot / len(va_i)

    history, best, best_state, best_epoch, bad = [], val_nll(), None, 0, 0
    history.append({'epoch': 0, 'train_nll': None, 'val_nll': best})
    best_state = copy.deepcopy(actor.state_dict())
    for epoch in range(1, cfg.max_epochs + 1):
        perm = tr_i[th.randperm(len(tr_i), generator=gen)]
        losses = []
        for batch in perm.split(cfg.batch_size):
            loss = _nll(actor, obs[batch], act[batch], cfg.action_clip)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = val_nll()
        history.append({'epoch': epoch, 'train_nll': float(np.mean(losses)), 'val_nll': v})
        log(f'BC epoch {epoch}: train NLL {np.mean(losses):.4f}, val NLL {v:.4f}')
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(actor.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    actor.load_state_dict(best_state)
    actor.set_training_mode(False)
    # --- report on the validation split ---
    with th.no_grad():
        mean, log_std, _ = actor.get_action_dist_params(obs[va_i])
        pred = th.tanh(mean).numpy()
        sigma = log_std.exp().numpy()
    vi = va_i.numpy()
    executed = d['actions'][vi]
    report = {'epochs_run': len(history) - 1, 'best_epoch': best_epoch, 'best_val_nll': best,
              'history': history, 'n_train_transitions': int(len(tr_i)), 'n_val_transitions': int(len(vi)),
              'train_episodes': train_eps.tolist(), 'val_episodes': val_eps.tolist(),
              'val_mse_executed_per_dim': ((pred - executed) ** 2).mean(0).tolist(),
              'mean_sigma_per_dim': sigma.mean(0).tolist(),
              'wall_s': time.perf_counter() - t0}
    if 'actions_clean' in d:
        report['val_mse_clean_per_dim'] = ((pred - d['actions_clean'][vi]) ** 2).mean(0).tolist()
    # A1 (D1_signoff): throttle vs max rear slip; the BC actor should reproduce the demo's
    # traction-control cut-off near 0.08-0.20.
    if d['obs'].shape[1] > SLIP_OBS.stop:
        slip = np.abs(d['obs'][d['obs_index'][vi]][:, SLIP_OBS]).max(1) / slip_gain
        edges = np.asarray(cfg.slip_bins, dtype=float)
        b = np.digitize(slip, edges) - 1
        rows = []
        for k in range(len(edges) - 1):
            m = b == k
            row = {'slip_lo': float(edges[k]), 'slip_hi': float(edges[k + 1]), 'n': int(m.sum())}
            if m.any():
                row['bc_throttle_pedal'] = float(pedal(pred[m, 1], pedal_deadzone).mean())
                row['executed_throttle_pedal'] = float(pedal(executed[m, 1], pedal_deadzone).mean())
                if 'actions_clean' in d:
                    row['demo_throttle_pedal'] = float(pedal(d['actions_clean'][vi][m, 1], pedal_deadzone).mean())
            rows.append(row)
        report['throttle_vs_rear_slip'] = rows
    return report


def plot_throttle_vs_slip(report, path):
    """A1 plot (PNG) from a train_bc report; skipped silently without matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = [r for r in report.get('throttle_vs_rear_slip', []) if r['n'] > 0]
    if not rows:
        return None
    x = [(r['slip_lo'] + r['slip_hi']) / 2 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r['bc_throttle_pedal'] for r in rows], 'o-', label='BC actor (deterministic)')
    if 'demo_throttle_pedal' in rows[0]:
        ax.plot(x, [r['demo_throttle_pedal'] for r in rows], 's--', label='demo driver (clean)')
    ax.set_xlabel('max |rear slip ratio|')
    ax.set_ylabel('mean throttle pedal')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
