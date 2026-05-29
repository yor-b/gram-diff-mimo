"""Channel estimation utilities."""

from __future__ import annotations

import numpy as np

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
    return np.linalg.solve(pilot_gram.T, matched_observations.T).T

def angular_pilot_observation(
    Y_p: np.ndarray,
    X_p: np.ndarray,
) -> np.ndarray:
    """Compute angular-domain pilot observation Y_tilde_p."""
    h_ls = least_squares_from_pilots(Y_p=Y_p, X_p=X_p)
    return np.fft.fft2(h_ls, norm="ortho")

def project_psd(matrix: np.ndarray) -> np.ndarray:
    """Project a Hermitian matrix onto the positive semidefinite cone."""
    hermitian = 0.5 * (matrix + matrix.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigenvalues_clipped = np.maximum(eigenvalues, 0.0)

    return eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.conj().T


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
    n_rx, n_data = Y_d.shape

    sample_cov = (Y_d @ Y_d.conj().T) / n_data
    gram_estimate = sample_cov - noise_variance * np.eye(n_rx)

    if project:
        gram_estimate = project_psd(gram_estimate)

    return gram_estimate


def angular_receive_gram(R_hat: np.ndarray) -> np.ndarray:
    """Transform receive-side Gram matrix estimate (R_hat) to angular domain.

    R_tilde = Phi_r R Phi_r^H
    """
    return np.fft.fft(
        np.fft.ifft(R_hat, axis=1, norm="ortho"),
        axis=0,
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