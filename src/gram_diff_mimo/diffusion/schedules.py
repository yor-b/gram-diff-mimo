"""Diffusion schedule utilities."""

from __future__ import annotations

import numpy as np


def alpha_bar_from_alpha(alpha: np.ndarray) -> np.ndarray:
    """Compute alpha_bar_t = product_{s=1}^t alpha_s."""
    return np.cumprod(alpha)


def diffusion_snr(alpha_bar: np.ndarray) -> np.ndarray:
    """Compute SNR_DM(t) = alpha_bar_t / (1 - alpha_bar_t)."""
    return alpha_bar / (1.0 - alpha_bar)


def snr_matched_timestep(
    observation_snr: float,
    alpha_bar: np.ndarray,
) -> int:
    """Return timestep whose diffusion SNR best matches observation SNR."""
    snr_dm = diffusion_snr(alpha_bar)
    return int(np.argmin(np.abs(snr_dm - observation_snr)))