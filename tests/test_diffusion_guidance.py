import numpy as np

from gram_diff_mimo.diffusion.guidance import (
    gram_guidance,
    likelihood_guidance,
    tweedie_clean_estimate,
)

from gram_diff_mimo.diffusion.guidance import ddim_denoise_step

from gram_diff_mimo.diffusion.sampler import (
    gram_diff_guided_sample,
    gram_diff_guided_step,
)


class ZeroNoiseDenoiser:
    def predict_noise(self, H_tilde_t, timestep):
        return np.zeros_like(H_tilde_t)

def test_tweedie_clean_estimate_zero_predicted_noise():
    H_tilde_t = np.ones((2, 2), dtype=np.complex128)
    predicted_noise = np.zeros_like(H_tilde_t)

    clean_estimate = tweedie_clean_estimate(
        H_tilde_t=H_tilde_t,
        predicted_noise=predicted_noise,
        alpha_bar_t=0.25,
    )

    assert np.allclose(clean_estimate, 2.0 * H_tilde_t)


def test_likelihood_guidance():
    Y_tilde = 3.0 * np.ones((2, 2), dtype=np.complex128)
    clean_estimate = np.ones((2, 2), dtype=np.complex128)

    guidance = likelihood_guidance(
        Y_tilde=Y_tilde,
        clean_estimate=clean_estimate,
        noise_variance=2.0,
    )

    assert np.allclose(guidance, np.ones((2, 2)))


def test_gram_guidance_zero_when_current_gram_matches_target():
    H_tilde_t = np.eye(2, dtype=np.complex128)
    R_tilde_hat = H_tilde_t @ H_tilde_t.conj().T

    guidance = gram_guidance(
        H_tilde_t=H_tilde_t,
        R_tilde_hat=R_tilde_hat,
    )

    assert np.allclose(guidance, np.zeros_like(H_tilde_t))


def test_gram_guidance_pushes_toward_larger_target_gram():
    H_tilde_t = np.eye(2, dtype=np.complex128)
    R_tilde_hat = 2.0 * np.eye(2, dtype=np.complex128)

    guidance = gram_guidance(
        H_tilde_t=H_tilde_t,
        R_tilde_hat=R_tilde_hat,
    )

    assert np.allclose(guidance, 4.0 * np.eye(2))

def test_ddim_denoise_step_with_zero_noise():
    H_tilde_t = np.ones((2, 2), dtype=np.complex128)
    predicted_noise = np.zeros_like(H_tilde_t)

    H_tilde_prev = ddim_denoise_step(
        H_tilde_t=H_tilde_t,
        predicted_noise=predicted_noise,
        alpha_bar_t=0.25,
        alpha_bar_prev=1.0,
    )

    assert np.allclose(H_tilde_prev, 2.0 * H_tilde_t)


def test_gram_diff_guided_step_reduces_to_ddim_when_guidance_zero():
    H_tilde_t = np.ones((2, 2), dtype=np.complex128)
    predicted_noise = np.zeros_like(H_tilde_t)
    Y_tilde = H_tilde_t.copy()
    R_tilde_hat = H_tilde_t @ H_tilde_t.conj().T

    H_prev = gram_diff_guided_step(
        H_tilde_t=H_tilde_t,
        Y_tilde=Y_tilde,
        R_tilde_hat=R_tilde_hat,
        predicted_noise=predicted_noise,
        alpha_bar_t=0.25,
        alpha_bar_prev=1.0,
        noise_variance=1.0,
        lambda_like=0.0,
        lambda_gram=0.0,
    )

    assert np.allclose(H_prev, 2.0 * H_tilde_t)


def test_gram_diff_guided_step_clips_gram_update_norm():
    H_tilde_t = np.eye(2, dtype=np.complex128)
    predicted_noise = np.zeros_like(H_tilde_t)
    Y_tilde = H_tilde_t.copy()
    R_tilde_hat = 2.0 * np.eye(2, dtype=np.complex128)

    H_prev = gram_diff_guided_step(
        H_tilde_t=H_tilde_t,
        Y_tilde=Y_tilde,
        R_tilde_hat=R_tilde_hat,
        predicted_noise=predicted_noise,
        alpha_bar_t=1.0,
        alpha_bar_prev=1.0,
        noise_variance=1.0,
        lambda_like=0.0,
        lambda_gram=1.0,
        gram_clip_norm=1.0,
    )

    assert np.isclose(np.linalg.norm(H_prev - H_tilde_t), 1.0)


def test_gram_diff_guided_sample_calls_denoiser_over_reverse_steps():
    H_tilde_start = np.ones((2, 2), dtype=np.complex128)
    alpha_bar = np.array([0.25])

    H_hat = gram_diff_guided_sample(
        H_tilde_start=H_tilde_start,
        Y_tilde=H_tilde_start.copy(),
        R_tilde_hat=H_tilde_start @ H_tilde_start.conj().T,
        denoiser=ZeroNoiseDenoiser(),
        alpha_bar=alpha_bar,
        noise_variance=1.0,
        t_start=0,
        lambda_like=0.0,
        lambda_gram=0.0,
    )

    assert np.allclose(H_hat, 2.0 * H_tilde_start)
