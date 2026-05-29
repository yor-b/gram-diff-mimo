from gram_diff_mimo.mimo.channel import (
    generate_rayleigh_channel,
    mimo_observation,
)

from gram_diff_mimo.mimo.metrics import nmse

import numpy as np


N_RX = 4
N_TX = 4
T = 16

noise_variance = 0.01


h = generate_rayleigh_channel(N_RX, N_TX)

# x = np.random.randn(N_TX, T) + 1j * np.random.randn(N_TX, T)
x = (np.random.randn(N_TX, T) + 1j * np.random.randn(N_TX, T)) / np.sqrt(2.0)

y, z = mimo_observation(
    h=h,
    x=x,
    noise_variance=noise_variance,
)

perfect_nmse = nmse(h, h)

print("H shape:", h.shape)
print("X shape:", x.shape)
print("Y shape:", y.shape)

print("Perfect NMSE:", perfect_nmse)