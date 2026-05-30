"""Channel estimation utilities."""

from __future__ import annotations

import numpy as np

from gram_diff_mimo.diffusion.denoiser import DiffusionNoisePredictor
from gram_diff_mimo.diffusion.sampler import gram_diff_guided_sample
from gram_diff_mimo.diffusion.schedules import snr_matched_timestep


def least_squares_from_pilots(
    Y_p: np.ndarray,
    X_p: np.ndarray,
) -> np.ndarray:
    """Least-squares channel estimate from pilot observations.

    Model:
        Y_p = H X_p + Z_p

    General LS:
        H_ls = Y_p X_p^H (X_p X_p^H)^(-1)

    If X_p X_p^H = I, this reduces to:
        H_ls = Y_p X_p^H
    """
    pilot_gram = X_p @ X_p.conj().T
    matched_observations = Y_p @ X_p.conj().T
    return np.linalg.solve(
        pilot_gram.T,
        matched_observations.swapaxes(-1, -2),
    ).swapaxes(-1, -2)

def angular_pilot_observation(
    Y_p: np.ndarray,
    X_p: np.ndarray,
) -> np.ndarray:
    """Compute angular-domain pilot observation Y_tilde_p."""
    h_ls = least_squares_from_pilots(Y_p=Y_p, X_p=X_p)
    return np.fft.fft2(h_ls, norm="ortho")

def project_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a Hermitian matrix onto the positive semidefinite cone."""
    hermitian = 0.5 * (matrix + matrix.conj().swapaxes(-1, -2))

    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigenvalues_clipped = np.maximum(eigenvalues, 0.0)

    return (
        (eigenvectors * eigenvalues_clipped[..., None, :])
        @ eigenvectors.conj().swapaxes(-1, -2)
    )


def estimate_receive_gram_from_data(
    Y_d: np.ndarray,
    noise_variance: float,
    project: bool = True,
) -> np.ndarray:
    """Estimate receive-side channel Gram matrix from data observations.

    Model:
        Y_d = H X_d + Z_d

    Approximation:
        (1 / N_d) Y_d Y_d^H ≈ H H^H + sigma^2 I

    Therefore:
        R_hat = (1 / N_d) Y_d Y_d^H - sigma^2 I
    """
    n_rx = Y_d.shape[-2]
    n_data = Y_d.shape[-1]

    sample_cov = (Y_d @ Y_d.conj().swapaxes(-1, -2)) / n_data
    gram_estimate = sample_cov - noise_variance * np.eye(n_rx)

    if project:
        gram_estimate = project_psd(gram_estimate)

    return gram_estimate


def angular_receive_gram(R_hat: np.ndarray) -> np.ndarray:
    """Transform receive-side Gram matrix estimate (R_hat) to angular domain.

    R_tilde = Phi_r R Phi_r^H
    """
    return np.fft.fft(
        np.fft.ifft(R_hat, axis=-1, norm="ortho"),
        axis=-2,
        norm="ortho",
    )

def estimate_angular_receive_gram_from_data(
    Y_d: np.ndarray,
    noise_variance: float,
    project: bool = True,
) -> np.ndarray:
    """Estimate angular-domain receive Gram matrix from data observations."""
    R_hat = estimate_receive_gram_from_data(
        Y_d=Y_d,
        noise_variance=noise_variance,
        project=project,
    )
    return angular_receive_gram(R_hat)

def variance_normalize_angular_observation(
    Y_tilde: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Initialize diffusion state from angular pilot observation.

    H_hat_tilde_tstar = (1 + sigma^2)^(-1/2) Y_tilde
    """
    return Y_tilde / np.sqrt(1.0 + noise_variance)


def diffusion_guidance_weights(
    betas: np.ndarray,
    *,
    lambda_like: float = 0.0,
    lambda_gram: float = 0.0,
    observation_snr: float | None = None,
    likelihood_gate_snr0: float | None = None,
    likelihood_gate_delta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-step guidance weights used by the GRAM-DIFF update."""
    betas = np.asarray(betas, dtype=np.float64)
    like_gate = 1.0
    if observation_snr is not None and likelihood_gate_snr0 is not None:
        like_gate = 1.0 / (
            1.0
            + np.exp(
                -(
                    observation_snr
                    - likelihood_gate_snr0
                )
                / likelihood_gate_delta
            )
        )

    like_weights = lambda_like * betas * like_gate
    gram_weights = lambda_gram * np.sqrt(betas)
    return like_weights, gram_weights


def gram_diff_channel_estimate(
    Y_p: np.ndarray,
    Y_d: np.ndarray | None,
    X_p: np.ndarray,
    noise_variance: float,
    denoiser: DiffusionNoisePredictor,
    alpha_bar: np.ndarray,
    betas: np.ndarray,
    *,
    lambda_like: float = 0.0,
    lambda_gram: float = 0.0,
    R_tilde_hat: np.ndarray | None = None,
    project_gram: bool = True,
    gram_clip_norm: float | np.ndarray | None = None,
    likelihood_gate_snr0: float | None = None,
    likelihood_gate_delta: float = 1.0,
) -> np.ndarray:
    """Estimate a MIMO channel using pretrained GRAM-DIFF.

    This is the end-to-end estimator from observations:
    pilot decorrelation, angular-domain SNR-matched initialization, optional
    data-aided Gram estimation, guided reverse diffusion, and inverse FFT.
    """
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive for SNR matching and likelihood guidance.")

    Y_tilde = angular_pilot_observation(Y_p=Y_p, X_p=X_p)
    H_tilde_start = variance_normalize_angular_observation(
        Y_tilde=Y_tilde,
        noise_variance=noise_variance,
    )

    if R_tilde_hat is None:
        if Y_d is None:
            R_tilde_shape = (*Y_p.shape[:-2], Y_p.shape[-2], Y_p.shape[-2])
            R_tilde_hat = np.zeros(R_tilde_shape, dtype=Y_p.dtype)
            lambda_gram = 0.0
        else:
            R_tilde_hat = estimate_angular_receive_gram_from_data(
                Y_d=Y_d,
                noise_variance=noise_variance,
                project=project_gram,
            )

    observation_snr = 1.0 / noise_variance
    t_start = snr_matched_timestep(
        observation_snr=observation_snr,
        alpha_bar=np.asarray(alpha_bar),
    )
    lambda_like_t, lambda_gram_t = diffusion_guidance_weights(
        betas=betas,
        lambda_like=lambda_like,
        lambda_gram=lambda_gram,
        observation_snr=observation_snr,
        likelihood_gate_snr0=likelihood_gate_snr0,
        likelihood_gate_delta=likelihood_gate_delta,
    )

    H_tilde_hat = gram_diff_guided_sample(
        H_tilde_start=H_tilde_start,
        Y_tilde=Y_tilde,
        R_tilde_hat=R_tilde_hat,
        denoiser=denoiser,
        alpha_bar=alpha_bar,
        noise_variance=noise_variance,
        t_start=t_start,
        lambda_like=lambda_like_t,
        lambda_gram=lambda_gram_t,
        gram_clip_norm=gram_clip_norm,
    )

    return np.fft.ifft2(H_tilde_hat, norm="ortho")
