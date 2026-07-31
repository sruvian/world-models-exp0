import os
import argparse
import itertools
from multiprocessing import Pool

from sim_envs import make_env
from collector import collect_trajectories

GRAVITIES = [5.0, 9.8, 15.0]
LENGTHS = [2.0, 10.0, 18.0]
SEEDS = list(range(43, 52))

NUM_TRAJECTORIES = 500
EPISODE_TIME = 1000
POLICY_SEED = 35

PENDULUM = {"mass1": 1.0, "mass2": 0.0, "dt": 0.01, "damping": 0.0}
CARTPOLE = {"mass1": 0.1, "mass2": 1.0, "dt": 0.01, "damping": 0.0}
IMPULSE_POLICY = [True, False]

ENVS = {
    "PendulumSim": ("pendulum", PENDULUM),
    "CartPoleSim": ("cartpole", CARTPOLE),
}


def build_jobs(top_dir):
    """One job per (env, gravity, length, seed, policy) combination."""
    jobs = []
    for env_name, (subdir, base_cfg) in ENVS.items():
        for g, l, seed, impulse in itertools.product(
            GRAVITIES, LENGTHS, SEEDS, IMPULSE_POLICY
        ):
            cfg = dict(base_cfg, gravity=g, length=l, seed=seed)
            out_sub = os.path.join(
                top_dir, "impulse_policy" if impulse else "", subdir
            )
            jobs.append((env_name, cfg, impulse, out_sub))
    return jobs


def run_one(job):

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    env_name, cfg, impulse, out_sub = job
    os.makedirs(out_sub, exist_ok=True)
    env = make_env(env_name, **cfg)
    collect_trajectories(
        env,
        num_trajectories=NUM_TRAJECTORIES,
        episode_time=EPISODE_TIME,
        policy_seed=POLICY_SEED,
        save=True,
        impulse_policy=impulse,
    )
    return f"{env_name} g={cfg['gravity']} l={cfg['length']} " \
           f"seed={cfg['seed']} impulse={impulse}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=None,
                    help="Parent directory for the 'datasets' folder. "
                         "Defaults to the current working directory.")
    ap.add_argument("--workers", type=int, default=14,
                    help="Number of parallel collection processes.")
    args = ap.parse_args()

    base = args.dataset_dir if args.dataset_dir else "."
    top_dir = os.path.join(base, "datasets")
    os.makedirs(top_dir, exist_ok=True)

    jobs = build_jobs(top_dir)
    print(f"Generating {len(jobs)} datasets with {args.workers} workers...")

    with Pool(processes=args.workers) as pool:
        for done in pool.imap_unordered(run_one, jobs):
            print(f"  done: {done}")

    print("All datasets generated.")