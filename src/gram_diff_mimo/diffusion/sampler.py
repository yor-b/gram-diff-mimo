"""GRAM-DIFF reverse diffusion sampler utilities."""

from __future__ import annotations

import numpy as np

from gram_diff_mimo.diffusion.denoiser import DiffusionNoisePredictor
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
    gram_clip_norm: float | None = None,
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
    gram_update = lambda_gram * g_gram
    if gram_clip_norm is not None:
        if gram_update.ndim == 2:
            gram_update_norm = np.linalg.norm(gram_update)
            if gram_update_norm > gram_clip_norm:
                gram_update = gram_update * (gram_clip_norm / (gram_update_norm + 1e-12))
        else:
            gram_update_norm = np.linalg.norm(
                gram_update.reshape(gram_update.shape[0], -1),
                axis=1,
            )
            scale = np.minimum(1.0, gram_clip_norm / (gram_update_norm + 1e-12))
            gram_update = gram_update * scale[:, None, None]

    return denoised_step + lambda_like * g_like + gram_update


def _scheduled_value(value: float | np.ndarray, timestep: int) -> float:
    if isinstance(value, np.ndarray):
        return float(value[timestep])
    return float(value)


def gram_diff_guided_sample(
    H_tilde_start: np.ndarray,
    Y_tilde: np.ndarray,
    R_tilde_hat: np.ndarray,
    denoiser: DiffusionNoisePredictor,
    alpha_bar: np.ndarray,
    noise_variance: float,
    *,
    t_start: int,
    lambda_like: float | np.ndarray = 0.0,
    lambda_gram: float | np.ndarray = 0.0,
    gram_clip_norm: float | np.ndarray | None = None,
) -> np.ndarray:
    """Run GRAM-DIFF reverse sampling with a pretrained epsilon predictor.

    Timesteps are zero-based and must match the indexing of ``alpha_bar`` and
    the pretrained denoiser. ``lambda_like`` and ``lambda_gram`` can be scalars
    or per-timestep arrays.
    """
    alpha_bar = np.asarray(alpha_bar, dtype=np.float64)
    if not 0 <= t_start < alpha_bar.shape[0]:
        raise ValueError(f"t_start must be in [0, {alpha_bar.shape[0] - 1}], got {t_start}.")

    H_tilde_t = np.asarray(H_tilde_start).copy()
    for timestep in range(t_start, -1, -1):
        predicted_noise = denoiser.predict_noise(H_tilde_t, timestep)
        alpha_bar_prev = 1.0 if timestep == 0 else float(alpha_bar[timestep - 1])
        H_tilde_t = gram_diff_guided_step(
            H_tilde_t=H_tilde_t,
            Y_tilde=Y_tilde,
            R_tilde_hat=R_tilde_hat,
            predicted_noise=predicted_noise,
            alpha_bar_t=float(alpha_bar[timestep]),
            alpha_bar_prev=alpha_bar_prev,
            noise_variance=noise_variance,
            lambda_like=_scheduled_value(lambda_like, timestep),
            lambda_gram=_scheduled_value(lambda_gram, timestep),
            gram_clip_norm=(
                None
                if gram_clip_norm is None
                else _scheduled_value(gram_clip_norm, timestep)
            ),
        )
    return H_tilde_t
