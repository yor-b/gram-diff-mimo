"""GRAM-DIFF reverse diffusion sampler utilities."""

from __future__ import annotations

import numpy as np

from gram_diff_mimo.diffusion.guidance import (
    ddim_denoise_step,
    gram_guidance,
    likelihood_guidance,
    tweedie_clean_estimate,
)


def gram_diff_guided_step(
    H_tilde_t: np.ndarray,
    Y_tilde: np.ndarray,
    R_tilde_hat: np.ndarray,
    predicted_noise: np.ndarray,
    alpha_bar_t: float,
    alpha_bar_prev: float,
    noise_variance: float,
    lambda_like: float,
    lambda_gram: float,
) -> np.ndarray:
    """Perform one GRAM-DIFF guided reverse step."""
    clean_estimate = tweedie_clean_estimate(
        H_tilde_t=H_tilde_t,
        predicted_noise=predicted_noise,
        alpha_bar_t=alpha_bar_t,
    )

    denoised_step = ddim_denoise_step(
        H_tilde_t=H_tilde_t,
        predicted_noise=predicted_noise,
        alpha_bar_t=alpha_bar_t,
        alpha_bar_prev=alpha_bar_prev,
    )

    g_like = likelihood_guidance(
        Y_tilde=Y_tilde,
        clean_estimate=clean_estimate,
        noise_variance=noise_variance,
    )

    g_gram = gram_guidance(
        H_tilde_t=H_tilde_t,
        R_tilde_hat=R_tilde_hat,
    )

    return denoised_step + lambda_like * g_like + lambda_gram * g_gram