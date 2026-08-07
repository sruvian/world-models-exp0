import os
import argparse
import itertools
from multiprocessing import Pool
import numpy as np
from sim_envs import make_env
from collector import collect_trajectories

GRAVITIES = [5.0, 9.8, 15.0]
LENGTHS = [2.0, 10.0, 18.0]
SEEDS = list(range(43, 52))

NUM_TRAJECTORIES = 500
EPISODE_TIME = 3000
POLICY_SEED = 35

PENDULUM = {"mass1": 1.0, "mass2": 0.0, "dt": 0.01, "damping": 0.0, 'max_action': 10}
CARTPOLE = {"mass1": 0.1, "mass2": 1.0, "dt": 0.01, "damping": 0.0, "max_action": 10}
IMPULSE_POLICY = [True, False]

ENVS = {
    "PendulumSim": ("pendulum", PENDULUM),
    "CartPoleSim": ("cartpole", CARTPOLE),
}

DRIVEN_CONFIGS = [(15.0, 2.0), (9.8, 2.0), (5.0, 2.0), (15.0, 0.5), (9.8, 1.0)]
DRIVE_AMP = 0.2
DAMPING = 0.5
N_OMEGA = 25
def build_driven_jobs(top_dir):
    jobs = []
    for (g, l), seed in zip(DRIVEN_CONFIGS, SEEDS):
        omega_0 = (g / l) ** 0.5
        omegas = np.linspace(0.5 * omega_0, 1.5 * omega_0, N_OMEGA)
        for w in omegas:
            cfg = dict(gravity=g, length=l, mass1=1.0, dt=0.01,
                       damping=DAMPING, max_action=0,
                       drive_omega=w, drive_amp=DRIVE_AMP, seed=seed)
            out_dir = os.path.join(top_dir, "driven")
            jobs.append(("DrivenPendulumSim", cfg, False, out_dir))
    return jobs

def build_jobs(top_dir):
    jobs = []
    configs = list(itertools.product(GRAVITIES, LENGTHS))
    for env_name, (subdir, base_cfg) in ENVS.items():
        for (g, l), seed in zip(configs, SEEDS):
            for impulse in IMPULSE_POLICY:
                cfg = dict(base_cfg, gravity=g, length=l, seed=seed)
                out_dir = os.path.join(top_dir, "impulse_policy" if impulse else "", subdir)
                jobs.append((env_name, cfg, impulse, out_dir))
    return jobs

def run_one(job):
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
    env_name, cfg, impulse, out_dir = job
    os.makedirs(out_dir, exist_ok=True)
    env = make_env(env_name, **cfg)
    collect_trajectories(env, num_trajectories=NUM_TRAJECTORIES, episode_time=EPISODE_TIME,
                         policy_seed=POLICY_SEED, save=True, path=out_dir, impulse_policy=impulse)
    return f"{env_name} g={cfg['gravity']} l={cfg['length']} seed={cfg['seed']} impulse={impulse}"


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

    # jobs = build_jobs(top_dir)
    # print(f"Generating {len(jobs)} datasets with {args.workers} workers...")

    # with Pool(processes=args.workers) as pool:
    #     for done in pool.imap_unordered(run_one, jobs):
    #         print(f"  done: {done}")
    driven_jobs = build_driven_jobs(top_dir)
    with Pool(processes=args.workers) as pool:
        for done in pool.imap_unordered(run_one, driven_jobs):
            print(f"  done: {done}")
    print("All datasets generated.")