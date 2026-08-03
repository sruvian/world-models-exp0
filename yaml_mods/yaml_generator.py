import yaml
import glob
import os
import copy
import argparse

SKIP_FILES = {"pendulum.yaml", "pendulum_test.yaml", "cartpole.yaml"}


def set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="mlp_seed_configs")
    parser.add_argument("--output_dir", default="combined_generated")

    parser.add_argument(
        "--latents",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64]
    )

    parser.add_argument(
        "--rollouts",
        nargs="+",
        type=int,
        default=[1, 3, 5, 15, 50]
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0]
    )
    

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    yaml_files = sorted(glob.glob(os.path.join(args.input_dir, "*.yaml")))

    total = 0

    for path in yaml_files:

        fname = os.path.basename(path)

        if fname in SKIP_FILES:
            print(f"Skipping {fname}")
            continue

        with open(path, "r") as f:
            base_cfg = yaml.safe_load(f)

        stem = os.path.splitext(fname)[0]

        for k in sorted(args.rollouts, reverse=True):

            for latent in args.latents:

                seed_list = args.seeds if args.seeds else [None]

                for seed in seed_list:

                    cfg = copy.deepcopy(base_cfg)

                    set_nested(cfg, "hyperparams.rollout_steps", k)
                    set_nested(cfg, "model.latent_dim", latent)

                    suffix = f"_k{k:03d}_lat{latent}"
                    if seed is not None:
                        set_nested(cfg, "model.seed", seed)
                        suffix += f"_seed{seed}"

                    out_name = stem + suffix + ".yaml"

                    with open(
                        os.path.join(args.output_dir, out_name),
                        "w",
                    ) as f:
                        yaml.safe_dump(
                            cfg,
                            f,
                            sort_keys=False,
                        )

                    total += 1

    print(f"\nGenerated {total} YAML files.")


if __name__ == "__main__":
    main()