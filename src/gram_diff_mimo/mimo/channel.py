"""Basic MIMO channel and observation utilities."""

from __future__ import annotations

import numpy as np


def complex_normal(shape: tuple[int, ...], variance: float = 1.0) -> np.ndarray:
    """Generate circularly symmetric complex Gaussian noise.

    Each complex entry has E[|z|^2] = variance.
    """
    std = np.sqrt(variance / 2.0)
    return std * (np.random.randn(*shape) + 1j * np.random.randn(*shape))


def generate_rayleigh_channel(n_rx: int, n_tx: int) -> np.ndarray:
    """Generate an i.i.d. Rayleigh fading MIMO channel.

    Entries satisfy E[|H_ij|^2] = 1.
    """
    return complex_normal((n_rx, n_tx), variance=1.0)

def mimo_observation(
    h: np.ndarray,
    x: np.ndarray,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy MIMO observations.

    Model:
        Y = HX + Z

    Parameters
    ----------
    h : np.ndarray
        Channel matrix of shape (n_rx, n_tx)

    x : np.ndarray
        Transmit signal matrix of shape (n_tx, n_symbols)

    noise_variance : float
        Complex noise variance E[|z|^2]

    Returns
    -------
    y : np.ndarray
        Received signal matrix

    z : np.ndarray
        Noise realization
    """
    z = complex_normal(
        (h.shape[0], x.shape[1]),
        variance=noise_variance,
    )

    y = h @ x + z

    return y, z