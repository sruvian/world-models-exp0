import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--log_dir", type=str, required=True)
    parser = args.parse_args()

    log_files = glob.glob(os.path.join(parser.log_dir, "*.npz"))
    
    if not log_files:
        print(f"No .npz files found in {parser.log_dir}")
        exit()

    print(f"Found {len(log_files)} log files")

    for log_path in log_files:
        try:
            if os.path.exists(log_path[:-4]+ ".png"):
                continue
            data = np.load(log_path, allow_pickle=True)
            train_loss = data["train_loss"]
            val_loss = data["val_loss"]
            steps = data["log_steps"]
            config = {k: data[k].item() for k in 
                      ["model_name", "optimizer_name", "loss_name", 
                       "lr", "batch_size", "steps", "gravity", "length", "latent_dim"]}

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, train_loss, label="Train Loss")
            ax.plot(steps, val_loss, label="Val Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(
                f"{config['model_name']} | g={config['gravity']} l={config['length']} "
                f"latent={config['latent_dim']} lr={config['lr']}"
            )
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            plot_path = log_path.replace(".npz", ".png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved: {os.path.basename(plot_path)}")

        except Exception as e:
            print(f"Failed {os.path.basename(log_path)}: {e}")

    print("Done.")