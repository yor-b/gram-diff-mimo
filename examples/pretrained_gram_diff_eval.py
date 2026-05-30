"""Run a small pretrained GRAM-DIFF channel-estimation smoke evaluation."""

from __future__ import annotations

import argparse

import numpy as np

from gram_diff_mimo.diffusion import load_fesl_pretrained_denoiser
from gram_diff_mimo.mimo.channel import generate_rayleigh_channel, mimo_observation
from gram_diff_mimo.mimo.estimators import (
    gram_diff_channel_estimate,
    least_squares_from_pilots,
)
from gram_diff_mimo.mimo.metrics import nmse
from gram_diff_mimo.mimo.pilots import identity_pilots


def random_qam_like_symbols(n_tx: int, n_symbols: int) -> np.ndarray:
    """Generate unit-power complex data symbols for Gram estimation."""
    return (
        np.random.randn(n_tx, n_symbols)
        + 1j * np.random.randn(n_tx, n_symbols)
    ) / np.sqrt(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="best_models_fesl_dm_paper/3gpp_path=3",
        help="Directory containing sim_params.json and train_models/*.pt.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--n-data", type=int, default=2000)
    parser.add_argument("--lambda-like", type=float, default=0.0)
    parser.add_argument("--lambda-gram", type=float, default=0.0)
    parser.add_argument("--gram-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--likelihood-gate-snr0",
        type=float,
        default=None,
        help="SNR transition point for likelihood guidance; omit to disable gating.",
    )
    parser.add_argument(
        "--likelihood-gate-delta",
        type=float,
        default=1.0,
        help="Smoothness of the likelihood-guidance SNR gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    denoiser = load_fesl_pretrained_denoiser(
        args.model_dir,
        device=args.device,
    )
    n_rx, n_tx = 64, 16
    H = generate_rayleigh_channel(n_rx=n_rx, n_tx=n_tx)

    X_p = identity_pilots(n_tx)
    Y_p, _ = mimo_observation(
        h=H,
        x=X_p,
        noise_variance=args.noise_variance,
    )

    X_d = random_qam_like_symbols(n_tx=n_tx, n_symbols=args.n_data)
    Y_d, _ = mimo_observation(
        h=H,
        x=X_d,
        noise_variance=args.noise_variance,
    )

    H_ls = least_squares_from_pilots(Y_p=Y_p, X_p=X_p)
    H_dm = gram_diff_channel_estimate(
        Y_p=Y_p,
        Y_d=None,
        X_p=X_p,
        noise_variance=args.noise_variance,
        denoiser=denoiser,
        alpha_bar=denoiser.alpha_bar,
        betas=denoiser.betas,
    )
    H_gram_diff = gram_diff_channel_estimate(
        Y_p=Y_p,
        Y_d=Y_d,
        X_p=X_p,
        noise_variance=args.noise_variance,
        denoiser=denoiser,
        alpha_bar=denoiser.alpha_bar,
        betas=denoiser.betas,
        lambda_like=args.lambda_like,
        lambda_gram=args.lambda_gram,
        gram_clip_norm=args.gram_clip_norm,
        likelihood_gate_snr0=args.likelihood_gate_snr0,
        likelihood_gate_delta=args.likelihood_gate_delta,
    )

    print(f"Model dir: {args.model_dir}")
    print(f"Device: {denoiser.device}")
    print(f"Noise variance: {args.noise_variance:g}")
    print(f"Data symbols: {args.n_data}")
    print(f"lambda_like: {args.lambda_like:g}")
    print(f"lambda_gram: {args.lambda_gram:g}")
    print(f"likelihood_gate_snr0: {args.likelihood_gate_snr0}")
    print(f"likelihood_gate_delta: {args.likelihood_gate_delta:g}")
    print(f"LS NMSE: {nmse(H, H_ls):.6g}")
    print(f"DM NMSE: {nmse(H, H_dm):.6g}")
    print(f"GRAM-DIFF NMSE: {nmse(H, H_gram_diff):.6g}")


if __name__ == "__main__":
    main()
