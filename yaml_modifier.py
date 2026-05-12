import yaml
import glob
import os
import argparse

SKIP_FILES = {"pendulum.yaml", "pendulum_test.yaml", "cartpole.yaml"}

CHANGES = {
    "hyperparams.rollout_steps": 50,
    # "hyperparams.rollout_decay": "Linear",
    # "datasets.use_existing": True,
    # "model.run_model": True,
    # "trainer.run_trainer": True,
    # "trainer.steps": 100000,
    # "checkpointing.save": True,
    # "checkpointing.load": False,
    # "rollout_engine.run_rollouts": False,
    # "model.latent_dim": 32,
}

def set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    old = d[keys[-1]]
    d[keys[-1]] = value
    return old

def patch_yaml_inplace(path, changes):
    """Edit yaml values in place using string replacement to preserve comments and formatting."""
    with open(path, 'r') as f:
        lines = f.readlines()

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    edits = []
    for dotted_key, new_val in changes.items():
        try:
            old_val = set_nested(config, dotted_key, new_val)
            edits.append((dotted_key.split(".")[-1], old_val, new_val))
        except KeyError as e:
            print(f"  WARNING: key not found {e}")
            continue  # add this

    for i, line in enumerate(lines):
        for leaf_key, old_val, new_val in edits:
            stripped = line.lstrip()
            if stripped.startswith(f"{leaf_key}:"):
                indent = line[: len(line) - len(stripped)]
                # preserve inline comments if any
                comment = ""
                val_part = stripped[len(leaf_key) + 1:].strip()
                if "#" in val_part:
                    comment = "  " + val_part[val_part.index("#"):]
                lines[i] = f"{indent}{leaf_key}: {format_val(new_val)}{comment}\n"
                break

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
    args.add_argument("--yaml_dir", type=str, default="configs/")
    args.add_argument("--dry_run", action="store_true")
    parser = args.parse_args()

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
                print(f"  {leaf}: {old} → {new}")
            if not parser.dry_run:
                with open(path, 'w') as f:
                    f.writelines(new_lines)
        else:
            print(f"No changes {fname}")

    if parser.dry_run:
        print("\n[DRY RUN] No files written.")