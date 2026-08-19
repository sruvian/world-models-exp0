import argparse, glob, os, copy, yaml

SKIP_FILES = {"pendulum.yaml", "pendulum_test.yaml", "cartpole.yaml"}

ARCH_OVERRIDES = {
    "mlp": {
        "model.name": "WorldModel",
        "model.hidden_dim": 64,
        "checkpointing.save_path": "model_saves_mlp",
        "checkpointing.logbase_dir": "logfiles_mlp",
    },
    "dmd": {
        "model.name": "WorldModelDMD",
        "model.hidden_dim": 128,
        "checkpointing.save_path": "model_saves_dmd",
        "checkpointing.logbase_dir": "logfiles_dmd",
    },
    "rssm": {
        "model.name": "WorldModelRSSM",
        "model.hidden_dim": 64,
        "checkpointing.save_path": "model_saves_rssm",
        "checkpointing.logbase_dir": "logfiles_rssm",
    },
    "gru": {
        "model.name": "WorldModelGRU",
        "model.hidden_dim": 64,
        "checkpointing.save_path": "model_saves_gru",
        "checkpointing.logbase_dir": "logfiles_gru",
    },
    "dmd_reg": {
        "model.name": "WorldModelDMD",
        "model.hidden_dim": 128,
        "hyperparams.reg": True,
        "checkpointing.save_path": "model_saves_dmd_reg",
        "checkpointing.logbase_dir": "logfiles_dmd_reg",
    },
}


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="cartpole_base")
    ap.add_argument("--output_dir", default="cartpole_generated")
    ap.add_argument("--archs", nargs="+", default=["mlp", "rssm", "dmd_reg"])
    ap.add_argument("--latents", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--rollouts", nargs="+", type=int, default=[1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    yaml_files = sorted(glob.glob(os.path.join(args.input_dir, "*.yaml")))
    total = 0

    for arch in args.archs:
        if arch not in ARCH_OVERRIDES:
            print(f"[warn] unknown arch '{arch}', skipping (known: {list(ARCH_OVERRIDES)})")
            continue

        for path in yaml_files:
            fname = os.path.basename(path)
            if fname in SKIP_FILES:
                print(f"Skipping {fname}")
                continue

            with open(path) as f:
                base_cfg = yaml.safe_load(f)
            stem = os.path.splitext(fname)[0]

            seed_list = args.seeds if args.seeds else [None]
            for seed in seed_list:                       # seeds
                for k in sorted(args.rollouts, reverse=True):   # rollouts
                    for latent in args.latents:          # latents
                        cfg = copy.deepcopy(base_cfg)

                        # per-arch stamps
                        for key, val in ARCH_OVERRIDES[arch].items():
                            set_nested(cfg, key, val)

                        set_nested(cfg, "hyperparams.rollout_steps", k)
                        set_nested(cfg, "model.latent_dim", latent)

                        suffix = f"_{arch}_k{k:03d}_lat{latent}"
                        if seed is not None:
                            set_nested(cfg, "model.seed", seed)
                            suffix += f"_seed{seed}"

                        out_name = stem + suffix + ".yaml"
                        with open(os.path.join(args.output_dir, out_name), "w") as f:
                            yaml.safe_dump(cfg, f, sort_keys=False)
                        total += 1

    print(f"\nGenerated {total} YAML files.")


if __name__ == "__main__":
    main()