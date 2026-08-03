import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.common.utils import parse_model, load_model
from analysis.common.data_provider import collect_for_config


EPS = 1e-9


def predicted_next(model, states: np.ndarray, actions: np.ndarray) -> np.ndarray:

    with torch.no_grad():
        st = torch.as_tensor(states, dtype=torch.float32)
        at = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, 1)

        encoded = model.encode(st)

        if isinstance(encoded, tuple):
            h, z = encoded

            if hasattr(model, "_join"):
                transition_input = model._join(h, z)
            else:
                transition_input = z

            next_latent = model.step(transition_input, at)
        else:
            next_latent = model.step(encoded, at)

        pred = model.decode(next_latent)

    return pred.detach().cpu().numpy()


def theta_of(states: np.ndarray) -> np.ndarray:
    return np.arctan2(states[:, 1], states[:, 0])


def make_pendulum_states(
    theta: np.ndarray,
    theta_dot: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            theta_dot,
        ]
    ).astype(np.float32)


def predicted_theta_ddot(
    states: np.ndarray,
    predicted_next_states: np.ndarray,
    dt: float,
) -> np.ndarray:
    return (
        predicted_next_states[:, 2] - states[:, 2]
    ) / dt


def true_theta_ddot(
    theta: np.ndarray,
    action: np.ndarray,
    gravity: float,
    length: float,
    mass: float,
    damping: float,
) -> np.ndarray:

    return (
        -(gravity / length) * np.sin(theta)
        + action / (mass * length**2)
    )


def overlap_bounds(
    values_a: np.ndarray,
    values_b: np.ndarray,
    quantile: float,
) -> tuple[float, float]:

    lower_a, upper_a = np.quantile(values_a, [quantile, 1.0 - quantile])
    lower_b, upper_b = np.quantile(values_b, [quantile, 1.0 - quantile])

    lower = max(lower_a, lower_b)
    upper = min(upper_a, upper_b)

    if lower >= upper:
        raise ValueError(
            "The two distributions have no robust overlapping support. "
            "Try a smaller --support_quantile or a more overlapping pair."
        )

    return float(lower), float(upper)


def normalized_position(
    prediction: np.ndarray,
    truth_a: np.ndarray,
    truth_b: np.ndarray,
) -> np.ndarray:
    difference = truth_b - truth_a

    alpha = np.full_like(prediction, np.nan, dtype=float)
    valid = np.abs(difference) > 1e-6

    alpha[valid] = (
        prediction[valid] - truth_a[valid]
    ) / difference[valid]

    return alpha


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)

    parser.add_argument("--gA", type=float, required=True)
    parser.add_argument("--lA", type=float, required=True)
    parser.add_argument("--gB", type=float, required=True)
    parser.add_argument("--lB", type=float, required=True)

    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--n_queries", type=int, default=5000)
    parser.add_argument("--n_traj", type=int, default=30)
    parser.add_argument("--steps", type=int, default=300)

    parser.add_argument(
        "--support_quantile",
        type=float,
        default=0.02,
        help=(
            "Discard this fraction from each tail before computing "
            "overlapping state-action support."
        ),
    )

    parser.add_argument(
        "--fixed_action",
        type=float,
        default=None,
        help=(
            "Use one identical action for every query. When omitted, actions "
            "are sampled uniformly from the overlapping action support."
        ),
    )

    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="overlap_test.png")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = parse_model(Path(args.ckpt))
    model = load_model(args.ckpt, cfg, args.device)
    model.eval()

    states_a, actions_a = collect_for_config(
        args.gA,
        args.lA,
        cfg["env"],
        cfg["impulse"],
        seed=args.seed,
        n_traj=args.n_traj,
        steps=args.steps,
    )

    states_b, actions_b = collect_for_config(
        args.gB,
        args.lB,
        cfg["env"],
        cfg["impulse"],
        seed=args.seed + 1,
        n_traj=args.n_traj,
        steps=args.steps,
    )

    states_a = np.asarray(states_a[:, :-1]).reshape(-1, states_a.shape[-1])
    states_b = np.asarray(states_b[:, :-1]).reshape(-1, states_b.shape[-1])

    actions_a = np.asarray(actions_a).reshape(-1)
    actions_b = np.asarray(actions_b).reshape(-1)

    theta_a = theta_of(states_a)
    theta_b = theta_of(states_b)

    theta_dot_a = states_a[:, 2]
    theta_dot_b = states_b[:, 2]

    theta_low, theta_high = overlap_bounds(
        theta_a,
        theta_b,
        args.support_quantile,
    )

    theta_dot_low, theta_dot_high = overlap_bounds(
        theta_dot_a,
        theta_dot_b,
        args.support_quantile,
    )

    theta_query = rng.uniform(
        theta_low,
        theta_high,
        size=args.n_queries,
    )

    theta_dot_query = rng.uniform(
        theta_dot_low,
        theta_dot_high,
        size=args.n_queries,
    )

    if args.fixed_action is not None:
        action_query = np.full(
            args.n_queries,
            args.fixed_action,
            dtype=np.float32,
        )
        action_low = action_high = args.fixed_action
    else:
        action_low, action_high = overlap_bounds(
            actions_a,
            actions_b,
            args.support_quantile,
        )

        action_query = rng.uniform(
            action_low,
            action_high,
            size=args.n_queries,
        ).astype(np.float32)

    common_states = make_pendulum_states(
        theta_query,
        theta_dot_query,
    )

    predicted_next_states = predicted_next(
        model,
        common_states,
        action_query,
    )

    pred_ddot = predicted_theta_ddot(
        common_states,
        predicted_next_states,
        args.dt,
    )

    true_ddot_a = true_theta_ddot(
        theta_query,
        action_query,
        args.gA,
        args.lA,
        args.mass,
        args.damping,
    ) - args.damping * theta_dot_query

    true_ddot_b = true_theta_ddot(
        theta_query,
        action_query,
        args.gB,
        args.lB,
        args.mass,
        args.damping,
    ) - args.damping * theta_dot_query

    midpoint_ddot = 0.5 * (true_ddot_a + true_ddot_b)

    alpha = normalized_position(
        pred_ddot,
        true_ddot_a,
        true_ddot_b,
    )

    valid_alpha = alpha[np.isfinite(alpha)]

    rmse_a = rmse(pred_ddot, true_ddot_a)
    rmse_b = rmse(pred_ddot, true_ddot_b)
    rmse_mid = rmse(pred_ddot, midpoint_ddot)

    mae_a = mae(pred_ddot, true_ddot_a)
    mae_b = mae(pred_ddot, true_ddot_b)
    mae_mid = mae(pred_ddot, midpoint_ddot)

    true_separation = np.abs(true_ddot_b - true_ddot_a)
    midpoint_error = np.abs(pred_ddot - midpoint_ddot)

    normalized_midpoint_error = np.median(
        midpoint_error / (true_separation + EPS)
    )

    print("\n" + "=" * 72)
    print("COMMON-INPUT CONFIGURATION TEST")
    print("=" * 72)

    print(
        f"Config A: g={args.gA:g}, l={args.lA:g}, "
        f"g/l={args.gA / args.lA:.4f}"
    )
    print(
        f"Config B: g={args.gB:g}, l={args.lB:g}, "
        f"g/l={args.gB / args.lB:.4f}"
    )

    print("\nCommon-query support:")
    print(f"  theta:     [{theta_low:.4f}, {theta_high:.4f}]")
    print(
        f"  theta_dot: [{theta_dot_low:.4f}, "
        f"{theta_dot_high:.4f}]"
    )
    print(f"  action:    [{action_low:.4f}, {action_high:.4f}]")
    print(f"  queries:   {args.n_queries}")

    print("\nPrediction error:")
    print(f"  RMSE to config A truth: {rmse_a:.6f}")
    print(f"  RMSE to config B truth: {rmse_b:.6f}")
    print(f"  RMSE to midpoint:       {rmse_mid:.6f}")

    print(f"  MAE to config A truth:  {mae_a:.6f}")
    print(f"  MAE to config B truth:  {mae_b:.6f}")
    print(f"  MAE to midpoint:        {mae_mid:.6f}")

    print("\nPosition between configuration truths:")
    if valid_alpha.size:
        print(f"  median alpha:            {np.median(valid_alpha):.4f}")
        print(
            f"  alpha IQR:               "
            f"[{np.quantile(valid_alpha, 0.25):.4f}, "
            f"{np.quantile(valid_alpha, 0.75):.4f}]"
        )
        print(
            f"  fraction A-like "
            f"(|alpha| < .25):           "
            f"{np.mean(np.abs(valid_alpha) < 0.25):.3f}"
        )
        print(
            f"  fraction midpoint-like "
            f"(|alpha-.5| < .25):        "
            f"{np.mean(np.abs(valid_alpha - 0.5) < 0.25):.3f}"
        )
        print(
            f"  fraction B-like "
            f"(|alpha-1| < .25):         "
            f"{np.mean(np.abs(valid_alpha - 1.0) < 0.25):.3f}"
        )
        print(
            f"  fraction outside [0,1]:   "
            f"{np.mean((valid_alpha < 0) | (valid_alpha > 1)):.3f}"
        )

    print(
        f"  normalized midpoint error: "
        f"{normalized_midpoint_error:.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].scatter(
        theta_a,
        theta_dot_a,
        s=3,
        alpha=0.25,
        label=(
            f"A: g={args.gA:g}, l={args.lA:g}, "
            f"g/l={args.gA / args.lA:.2f}"
        ),
    )

    axes[0].scatter(
        theta_b,
        theta_dot_b,
        s=3,
        alpha=0.25,
        label=(
            f"B: g={args.gB:g}, l={args.lB:g}, "
            f"g/l={args.gB / args.lB:.2f}"
        ),
    )

    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$\dot{\theta}$")
    axes[0].set_title("Natural phase-space coverage")
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(
        true_ddot_a,
        true_ddot_b,
        c=np.clip(alpha, 0, 1),
        s=8,
        alpha=0.6,
    )

    axes[1].set_xlabel(r"True $\ddot{\theta}$ under A")
    axes[1].set_ylabel(r"True $\ddot{\theta}$ under B")
    axes[1].set_title("Shared-input dynamical ambiguity")

    colorbar = fig.colorbar(scatter, ax=axes[1])
    colorbar.set_label(r"Model position $\alpha$: A $\rightarrow$ B")

    axes[2].scatter(
        midpoint_ddot,
        pred_ddot,
        s=8,
        alpha=0.5,
    )

    limits = [
        min(midpoint_ddot.min(), pred_ddot.min()),
        max(midpoint_ddot.max(), pred_ddot.max()),
    ]

    axes[2].plot(
        limits,
        limits,
        linestyle="--",
        linewidth=1,
        label="midpoint prediction",
    )

    axes[2].set_xlabel(r"Midpoint of true $\ddot{\theta}$")
    axes[2].set_ylabel(r"Model-predicted $\hat{\ddot{\theta}}$")
    axes[2].set_title(
        f"Common-input prediction\n"
        f"median alpha={np.nanmedian(alpha):.3f}"
    )
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        args.out,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"\nSaved plot to: {args.out}")


if __name__ == "__main__":
    main()