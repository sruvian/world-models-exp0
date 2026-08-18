import yaml
import glob
import os
import argparse

SKIP_FILES = {"pendulum.yaml", "pendulum_test.yaml", "cartpole.yaml"}

CHANGES = {
    # # "hyperparams.rollout_steps": 50,
    # # "hyperparams.rollout_decay": "Linear",
    # "datasets.use_existing": True,
    # "model.run_model": True,
    # "trainer.run_trainer": True,
    # # "trainer.steps": 100000,
    # "checkpointing.save": True,
    # # "collector.num_trajectories": 500,
    # # "checkpointing.load": False,
    # # "rollout_engine.run_rollouts": False,
    # # "model.latent_dim": 32,
    # "collector.save": False,
    # "collector.impulse_policy": True
    # "settings.device": "cpu"
    # "model.seed": 100
    # "model.hidden_dim": 64
    # "checkpointing.save_path": "model_saves_rssm",
    # "checkpointing.logbase_dir": "logfiles_rssm",
    # "model.name": "WorldModelRSSM"
    # "datasets.windowed_cache": None,
    # "hyperparams.transient": 0,
    # "hyperparams.windows_per_traj": 1
}

def set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    old = d.get(keys[-1], "__missing__")
    d[keys[-1]] = value
    return old

def patch_yaml_inplace(path, changes):
    with open(path, 'r') as f:
        lines = f.readlines()
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    edits = []
    for dotted_key, new_val in changes.items():
        try:
            old_val = set_nested(config, dotted_key, new_val)
            edits.append((dotted_key, old_val, new_val))
        except KeyError as e:
            print(f"  WARNING: key not found {e}")
            continue

    for dotted_key, old_val, new_val in edits:
        parts = dotted_key.split(".")
        section = parts[-2] if len(parts) > 1 else None
        leaf_key = parts[-1]

        in_section = False
        replaced = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if section and line.strip() == f"{section}:":
                in_section = True
                continue
            if in_section and not line.startswith(" ") and line.strip().endswith(":"):
                in_section = False
            if (in_section or section is None) and stripped.startswith(f"{leaf_key}:"):
                indent = line[: len(line) - len(stripped)]
                comment = ""
                val_part = stripped[len(leaf_key) + 1:].strip()
                if "#" in val_part:
                    comment = "  " + val_part[val_part.index("#"):]
                lines[i] = f"{indent}{leaf_key}: {format_val(new_val)}{comment}\n"
                replaced = True
                break

        if not replaced:
            if section is None:
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] += "\n"
                lines.append(f"{leaf_key}: {format_val(new_val)}\n")
            else:
                inserted = False
                for i, line in enumerate(lines):
                    if line.strip() == f"{section}:":
                        indent = "  "
                        for j in range(i + 1, len(lines)):
                            nxt = lines[j]
                            if nxt.strip() == "":
                                continue
                            child_indent = nxt[: len(nxt) - len(nxt.lstrip())]
                            if child_indent:
                                indent = child_indent
                            break
                        lines.insert(i + 1, f"{indent}{leaf_key}: {format_val(new_val)}\n")
                        inserted = True
                        break
                if not inserted:
                    print(f"  WARNING: section '{section}' not found for {dotted_key}, skipped")

    return lines, edits

def format_val(val):
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, str):
        return f'"{val}"'
    if val is None:
        return "null"
    return str(val)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--yaml_dir", type=str, default="trainer_configs/")
    args.add_argument("--dry_run", action="store_true")
    args.add_argument("--latent", type=int, default=None)
    args.add_argument("--k", type=int, default=None)
    args.add_argument("--seed", type=int, default=None)
    args.add_argument('--beta', type = float, default = None)
    parser = args.parse_args()

    # build CHANGES dynamically from CLI args
    if parser.latent is not None:
        CHANGES["model.latent_dim"] = parser.latent
    if parser.k is not None:
        CHANGES["hyperparams.rollout_steps"] = parser.k
    if parser.beta is not None:
        CHANGES["hyperparams.beta"] = parser.beta
    if parser.seed is not None:
        CHANGES["model.seed"] = parser.seed
    if not CHANGES:
        print("No changes configured.")
        exit()

    yaml_files = glob.glob(os.path.join(parser.yaml_dir, "*.yaml"))


    for path in yaml_files:
        fname = os.path.basename(path)
        if fname in SKIP_FILES:
            print(f"Skipped  {fname} (artifact)")
            continue

        new_lines, edits = patch_yaml_inplace(path, CHANGES)

        changed = [(k, o, n) for k, o, n in edits if o != n]
        if changed:
            print(f"\n{fname}:")
            for leaf, old, new in changed:
                print(f"  {leaf}: {old} -> {new}")
            if not parser.dry_run:
                with open(path, 'w') as f:
                    f.writelines(new_lines)
        else:
            print(f"No changes {fname}")

    if parser.dry_run:
        print("\n[DRY RUN] No files written.")