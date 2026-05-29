"""Interfaces for pretrained diffusion noise predictors."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DiffusionNoisePredictor(Protocol):
    """Interface for a pretrained diffusion model epsilon_theta."""

    def predict_noise(
        self,
        H_tilde_t: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """Predict diffusion noise epsilon_theta(H_tilde_t, t)."""
        ...