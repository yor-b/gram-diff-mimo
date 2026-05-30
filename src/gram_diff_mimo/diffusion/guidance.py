"""Guidance terms for GRAM-DIFF reverse diffusion."""

from __future__ import annotations

import numpy as np


def tweedie_clean_estimate(
    H_tilde_t: np.ndarray,
    predicted_noise: np.ndarray,
    alpha_bar_t: float,
) -> np.ndarray:
    """Compute T(H_tilde_t), the denoised clean-channel estimate."""
    return (
        H_tilde_t
        - np.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / np.sqrt(alpha_bar_t)


def likelihood_guidance(
    Y_tilde: np.ndarray,
    clean_estimate: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Compute likelihood guidance direction."""
    return (Y_tilde - clean_estimate) / noise_variance


def gram_guidance(
    H_tilde_t: np.ndarray,
    R_tilde_hat: np.ndarray,
) -> np.ndarray:
    """Compute Gram matrix guidance direction."""
    current_gram = H_tilde_t @ H_tilde_t.conj().swapaxes(-1, -2)
    return 4.0 * (R_tilde_hat - current_gram) @ H_tilde_t

def ddim_denoise_step(
    H_tilde_t: np.ndarray,
    predicted_noise: np.ndarray,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> np.ndarray:
    """Compute deterministic DDIM denoising step D_t(H_tilde_t)."""
    clean_estimate = tweedie_clean_estimate(
        H_tilde_t=H_tilde_t,
        predicted_noise=predicted_noise,
        alpha_bar_t=alpha_bar_t,
    )

    return (
        np.sqrt(alpha_bar_prev) * clean_estimate
        + np.sqrt(1.0 - alpha_bar_prev) * predicted_noise
    )
