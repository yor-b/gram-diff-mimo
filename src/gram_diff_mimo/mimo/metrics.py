"""Metrics for MIMO channel estimation."""

from __future__ import annotations

import numpy as np


def nmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute normalized mean-squared error.

    NMSE = ||reference - estimate||_F^2 / ||reference||_F^2
    """
    numerator = np.linalg.norm(reference - estimate, ord="fro") ** 2
    denominator = np.linalg.norm(reference, ord="fro") ** 2

    if denominator == 0:
        raise ValueError("NMSE is undefined when the reference has zero norm.")

    return float(numerator / denominator)