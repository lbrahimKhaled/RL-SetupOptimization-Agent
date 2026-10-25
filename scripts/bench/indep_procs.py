"""Throwaway (sim-runner): N independent processes, each DrivingEnv + random actions (no vec env,
no lockstep): CPU scaling ceiling."""
import json, sys, time, multiprocessing as mp
sys.path.insert(0, '/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent')


def work(args):
    rank, n_dec, barrier = args
    import numpy as np
    from rlsetup.driving.env import DrivingEnv
    from rlsetup.driving.config import EnvConfig
    env = DrivingEnv(EnvConfig())
    rng = np.random.default_rng(rank)
    env.reset(seed=rank)
    barrier.wait()
    t = time.perf_counter(); n = 0; t_step = 0.
    while n < n_dec:
        a = rng.uniform(-1, 1, 3).astype(np.float32)
        t1 = time.perf_counter()
        _, _, te, tr, _ = env.step(a)
        t_step += time.perf_counter() - t1
        n += 1
        if te or tr:
            env.reset(seed=int(rng.integers(2**31)))
    return n / (time.perf_counter() - t), n / t_step


if __name__ == '__main__':
    from ocd_lap.track import Track
    from rlsetup.driving import qss_profile
    qss_profile.load_or_build(Track('Shanghai'))
    ctx = mp.get_context('forkserver')
    for n in [int(x) for x in sys.argv[1:]] or [1, 4, 8]:
        with ctx.Manager() as m:
            b = m.Barrier(n)
            with ctx.Pool(n) as pool:
                t = time.perf_counter()
                res = pool.map(work, [(i, 4000, b) for i in range(n)])
                wall = time.perf_counter() - t
        incl = [r[0] for r in res]; step = [r[1] for r in res]
        print(json.dumps(dict(procs=n, sum_decisions_per_s_incl_resets=sum(incl), min_proc=min(incl),
                              max_proc=max(incl), sum_step_only=sum(step))), flush=True)
