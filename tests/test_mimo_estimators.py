import numpy as np

from gram_diff_mimo.mimo.channel import generate_rayleigh_channel, mimo_observation
from gram_diff_mimo.mimo.estimators import (
    angular_pilot_observation,
    angular_receive_gram,
    estimate_receive_gram_from_data,
    least_squares_from_pilots,
    variance_normalize_angular_observation
)
from gram_diff_mimo.mimo.pilots import identity_pilots
from gram_diff_mimo.diffusion.schedules import (
alpha_bar_from_alpha,
snr_matched_timestep,
)

def test_least_squares_recovers_channel_with_identity_pilots_zero_noise():
    H = generate_rayleigh_channel(4, 4)
    X_p = identity_pilots(4)
    Y_p, _ = mimo_observation(H, X_p, noise_variance=0.0)

    H_ls = least_squares_from_pilots(Y_p, X_p)

    assert np.allclose(H_ls, H)


def test_angular_pilot_observation_matches_fft_of_channel_zero_noise():
    H = generate_rayleigh_channel(4, 4)
    X_p = identity_pilots(4)
    Y_p, _ = mimo_observation(H, X_p, noise_variance=0.0)

    Y_tilde = angular_pilot_observation(Y_p, X_p)

    assert np.allclose(Y_tilde, np.fft.fft2(H, norm="ortho"))


def test_angular_receive_gram_matches_angular_channel_gram():
    H = generate_rayleigh_channel(8, 4)

    H_tilde = np.fft.fft2(H, norm="ortho")
    R = H @ H.conj().T

    R_tilde_from_R = angular_receive_gram(R)
    R_tilde_from_H_tilde = H_tilde @ H_tilde.conj().T

    assert np.allclose(R_tilde_from_R, R_tilde_from_H_tilde)


def test_receive_gram_estimator_converges_with_many_data_symbols():
    N_r = 8
    N_t = 4
    N_d = 100_000

    H = generate_rayleigh_channel(N_r, N_t)

    X_d = (
        np.random.randn(N_t, N_d)
        + 1j * np.random.randn(N_t, N_d)
    ) / np.sqrt(2.0)

    Y_d, _ = mimo_observation(H, X_d, noise_variance=0.0)

    R_true = H @ H.conj().T
    R_hat = estimate_receive_gram_from_data(Y_d, noise_variance=0.0)

    rel_error = np.linalg.norm(R_true - R_hat, ord="fro") / np.linalg.norm(
        R_true,
        ord="fro",
    )

    assert rel_error < 0.02


def test_snr_matching():
    alpha = np.array([0.9, 0.8, 0.7])

    alpha_bar = alpha_bar_from_alpha(alpha)

    observation_snr = (
        alpha_bar[1]
        / (1.0 - alpha_bar[1])
    )

    t = snr_matched_timestep(
        observation_snr=observation_snr,
        alpha_bar=alpha_bar,
    )

    assert t == 1

def test_variance_normalize_angular_observation():
    Y_tilde = np.ones((2, 2), dtype=np.complex128)

    H_tilde_tstar = variance_normalize_angular_observation(
        Y_tilde,
        noise_variance=3.0,
    )

    assert np.allclose(
        H_tilde_tstar,
        0.5 * Y_tilde,
    )