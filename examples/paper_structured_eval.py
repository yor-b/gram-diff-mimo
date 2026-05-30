"""Paper-structured GRAM-DIFF evaluation harness.

This compares the estimator variants used in the GRAM-DIFF paper while keeping
the paper's guidance schedule forms fixed:

    lambda_like,t = lambda_like * beta_t * SNR_gate
    lambda_gram,t = lambda_gram * sqrt(beta_t)

The channel generator here is the repo's simple Rayleigh simulator, so this is a
sanity/tuning harness rather than a reproduction of the paper's 3GPP or QuaDRiGa
experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gram_diff_mimo.diffusion import load_fesl_pretrained_denoiser
from gram_diff_mimo.mimo.channel import generate_rayleigh_channels, mimo_observation
from gram_diff_mimo.mimo.estimators import (
    angular_receive_gram,
    gram_diff_channel_estimate,
    least_squares_from_pilots,
)
from gram_diff_mimo.mimo.metrics import nmse_per_sample
from gram_diff_mimo.mimo.pilots import identity_pilots


VARIANT_STYLES = {
    "LS": {"linestyle": "--", "marker": "o"},
    "DM": {"linestyle": "-", "marker": "s"},
    "DM+Likelihood": {"linestyle": "-", "marker": "^"},
    "DM+Gram(est)": {"linestyle": "-", "marker": "D"},
    "DM+Gram(oracle)": {"linestyle": ":", "marker": "D"},
    "Joint(est)": {"linestyle": "-", "marker": "v"},
    "Joint(oracle)": {"linestyle": ":", "marker": "v"},
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def snr_grid_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.snr_db is not None:
        return np.asarray(parse_csv_floats(args.snr_db), dtype=np.float64)
    if args.n_snr_points < 1:
        raise ValueError("--n-snr-points must be at least 1.")
    return np.linspace(args.snr_min_db, args.snr_max_db, args.n_snr_points)


def data_symbols(batch_size: int, n_tx: int, n_data: int) -> np.ndarray:
    return (
        np.random.randn(batch_size, n_tx, n_data)
        + 1j * np.random.randn(batch_size, n_tx, n_data)
    ) / np.sqrt(2.0)


def rows_for_n_data(
    rows: list[dict[str, str | int | float]],
    n_data: int,
) -> list[dict[str, str | int | float]]:
    return [row for row in rows if row["n_data"] == n_data]


def plot_nmse_vs_snr(
    rows: list[dict[str, str | int | float]],
    n_data: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    n_data_rows = rows_for_n_data(rows, n_data)
    variants = [variant for variant in VARIANT_STYLES if any(row["variant"] == variant for row in n_data_rows)]

    for variant in variants:
        variant_rows = sorted(
            [row for row in n_data_rows if row["variant"] == variant],
            key=lambda row: float(row["snr_db"]),
        )
        x = [float(row["snr_db"]) for row in variant_rows]
        y = [float(row["mean_nmse"]) for row in variant_rows]
        ax.semilogy(x, y, label=variant, **VARIANT_STYLES[variant])

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE")
    ax.set_title(f"NMSE vs SNR, Nd={n_data}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def estimate_variants(
    *,
    H: np.ndarray,
    Y_p: np.ndarray,
    Y_d: np.ndarray,
    X_p: np.ndarray,
    noise_variance: float,
    denoiser,
    lambda_like: float,
    lambda_gram: float,
    gram_clip_norm: float,
    likelihood_gate_snr0: float | None,
    likelihood_gate_delta: float,
) -> dict[str, np.ndarray]:
    R_tilde_oracle = angular_receive_gram(H @ H.conj().swapaxes(-1, -2))

    variants: dict[str, np.ndarray] = {}
    variants["LS"] = nmse_per_sample(H, least_squares_from_pilots(Y_p=Y_p, X_p=X_p))

    H_dm = gram_diff_channel_estimate(
        Y_p=Y_p,
        Y_d=None,
        X_p=X_p,
        noise_variance=noise_variance,
        denoiser=denoiser,
        alpha_bar=denoiser.alpha_bar,
        betas=denoiser.betas,
    )
    variants["DM"] = nmse_per_sample(H, H_dm)

    if lambda_like == 0.0:
        variants["DM+Likelihood"] = variants["DM"].copy()
    else:
        H_like = gram_diff_channel_estimate(
            Y_p=Y_p,
            Y_d=None,
            X_p=X_p,
            noise_variance=noise_variance,
            denoiser=denoiser,
            alpha_bar=denoiser.alpha_bar,
            betas=denoiser.betas,
            lambda_like=lambda_like,
            likelihood_gate_snr0=likelihood_gate_snr0,
            likelihood_gate_delta=likelihood_gate_delta,
        )
        variants["DM+Likelihood"] = nmse_per_sample(H, H_like)

    if lambda_gram == 0.0:
        variants["DM+Gram(est)"] = variants["DM"].copy()
        variants["DM+Gram(oracle)"] = variants["DM"].copy()
    else:
        H_gram_est = gram_diff_channel_estimate(
            Y_p=Y_p,
            Y_d=Y_d,
            X_p=X_p,
            noise_variance=noise_variance,
            denoiser=denoiser,
            alpha_bar=denoiser.alpha_bar,
            betas=denoiser.betas,
            lambda_gram=lambda_gram,
            gram_clip_norm=gram_clip_norm,
        )
        variants["DM+Gram(est)"] = nmse_per_sample(H, H_gram_est)

        H_gram_oracle = gram_diff_channel_estimate(
            Y_p=Y_p,
            Y_d=None,
            X_p=X_p,
            noise_variance=noise_variance,
            denoiser=denoiser,
            alpha_bar=denoiser.alpha_bar,
            betas=denoiser.betas,
            lambda_gram=lambda_gram,
            R_tilde_hat=R_tilde_oracle,
            gram_clip_norm=gram_clip_norm,
        )
        variants["DM+Gram(oracle)"] = nmse_per_sample(H, H_gram_oracle)

    if lambda_like == 0.0:
        variants["Joint(est)"] = variants["DM+Gram(est)"].copy()
        variants["Joint(oracle)"] = variants["DM+Gram(oracle)"].copy()
    elif lambda_gram == 0.0:
        variants["Joint(est)"] = variants["DM+Likelihood"].copy()
        variants["Joint(oracle)"] = variants["DM+Likelihood"].copy()
    else:
        H_joint_est = gram_diff_channel_estimate(
            Y_p=Y_p,
            Y_d=Y_d,
            X_p=X_p,
            noise_variance=noise_variance,
            denoiser=denoiser,
            alpha_bar=denoiser.alpha_bar,
            betas=denoiser.betas,
            lambda_like=lambda_like,
            lambda_gram=lambda_gram,
            gram_clip_norm=gram_clip_norm,
            likelihood_gate_snr0=likelihood_gate_snr0,
            likelihood_gate_delta=likelihood_gate_delta,
        )
        variants["Joint(est)"] = nmse_per_sample(H, H_joint_est)

        H_joint_oracle = gram_diff_channel_estimate(
            Y_p=Y_p,
            Y_d=None,
            X_p=X_p,
            noise_variance=noise_variance,
            denoiser=denoiser,
            alpha_bar=denoiser.alpha_bar,
            betas=denoiser.betas,
            lambda_like=lambda_like,
            lambda_gram=lambda_gram,
            R_tilde_hat=R_tilde_oracle,
            gram_clip_norm=gram_clip_norm,
            likelihood_gate_snr0=likelihood_gate_snr0,
            likelihood_gate_delta=likelihood_gate_delta,
        )
        variants["Joint(oracle)"] = nmse_per_sample(H, H_joint_oracle)

    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="best_models_fesl_dm_paper/3gpp_path=3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--snr-db",
        default=None,
        help="Optional comma-separated SNR values in dB. Overrides the linspace grid.",
    )
    parser.add_argument("--snr-min-db", type=float, default=-15.0)
    parser.add_argument("--snr-max-db", type=float, default=15.0)
    parser.add_argument("--n-snr-points", type=int, default=31)
    parser.add_argument("--n-data", default="200,2000")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lambda-like", type=float, default=0.0)
    parser.add_argument("--lambda-gram", type=float, default=0.01)
    parser.add_argument("--gram-clip-norm", type=float, default=0.5)
    parser.add_argument("--likelihood-gate-snr0", type=float, default=None)
    parser.add_argument("--likelihood-gate-delta", type=float, default=1.0)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar for batched evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    np.random.seed(args.seed)

    snr_db_values = snr_grid_from_args(args)
    n_data_values = parse_csv_ints(args.n_data)
    print(f"Loading denoiser from {args.model_dir}...", flush=True)
    denoiser = load_fesl_pretrained_denoiser(args.model_dir, device=args.device)

    rows: list[dict[str, str | int | float]] = []
    n_rx, n_tx = 64, 16
    X_p = identity_pilots(n_tx)
    total_batches = (
        len(snr_db_values)
        * len(n_data_values)
        * ((args.n_trials + args.batch_size - 1) // args.batch_size)
    )
    print(
        f"Evaluating {len(snr_db_values)} SNR point(s), {len(n_data_values)} data setting(s), "
        f"{args.n_trials} trial(s) each on {denoiser.device}. Total batches: {total_batches}.",
        flush=True,
    )

    with tqdm(total=total_batches, unit="batch", disable=not args.progress, dynamic_ncols=True) as progress:
        for snr_db in snr_db_values:
            noise_variance = 10.0 ** (-snr_db / 10.0)
            for n_data in n_data_values:
                totals: dict[str, list[float]] = {}
                for start in range(0, args.n_trials, args.batch_size):
                    current_batch_size = min(args.batch_size, args.n_trials - start)
                    H = generate_rayleigh_channels(
                        batch_size=current_batch_size,
                        n_rx=n_rx,
                        n_tx=n_tx,
                    )
                    Y_p, _ = mimo_observation(
                        h=H,
                        x=X_p,
                        noise_variance=noise_variance,
                    )
                    X_d = data_symbols(
                        batch_size=current_batch_size,
                        n_tx=n_tx,
                        n_data=n_data,
                    )
                    Y_d, _ = mimo_observation(
                        h=H,
                        x=X_d,
                        noise_variance=noise_variance,
                    )
                    trial = estimate_variants(
                        H=H,
                        Y_p=Y_p,
                        Y_d=Y_d,
                        X_p=X_p,
                        noise_variance=noise_variance,
                        denoiser=denoiser,
                        lambda_like=args.lambda_like,
                        lambda_gram=args.lambda_gram,
                        gram_clip_norm=args.gram_clip_norm,
                        likelihood_gate_snr0=args.likelihood_gate_snr0,
                        likelihood_gate_delta=args.likelihood_gate_delta,
                    )
                    for name, value in trial.items():
                        totals.setdefault(name, []).extend(value.tolist())
                    progress.update(1)

                for name, values in totals.items():
                    row = {
                        "snr_db": snr_db,
                        "n_data": n_data,
                        "variant": name,
                        "mean_nmse": float(np.mean(values)),
                        "std_nmse": float(np.std(values)),
                        "n_trials": args.n_trials,
                    }
                    rows.append(row)
                    tqdm.write(
                        f"SNR={snr_db:>5g} dB  Nd={n_data:>5d}  "
                        f"{name:<15} mean={row['mean_nmse']:.6g} std={row['std_nmse']:.3g}"
                    )

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["snr_db", "n_data", "variant", "mean_nmse", "std_nmse", "n_trials"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")

    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        for n_data in n_data_values:
            output_path = plot_dir / f"nmse_vs_snr_nd_{n_data}.png"
            plot_nmse_vs_snr(rows, n_data=n_data, output_path=output_path)
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
