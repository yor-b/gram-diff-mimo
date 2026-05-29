"""Pilot generation utilities for MIMO systems."""

from __future__ import annotations

import numpy as np


def identity_pilots(n_tx: int) -> np.ndarray:
    """Generate orthogonal identity pilots.

    Shape:
        (n_tx, n_tx)

    This gives X_p X_p^H = I.
    """
    return np.eye(n_tx, dtype=np.complex128)

