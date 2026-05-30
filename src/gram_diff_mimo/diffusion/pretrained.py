"""Adapters for pretrained diffusion-channel-estimation checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local env
    torch = None


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on local env
        raise ImportError(
            "PyTorch is required to load pretrained diffusion checkpoints. "
            "Install torch in the active project environment first."
        )
    return torch


_TorchModule = torch.nn.Module if torch is not None else object


class _FeslCNN(_TorchModule):
    """Lightweight CNN used by Fesl et al.'s DM channel estimator."""

    def __init__(
        self,
        *,
        data_shape: list[int] | tuple[int, ...],
        n_layers_pre: int,
        n_layers_post: int,
        ch_layers_pre: list[int] | tuple[int, ...],
        ch_layers_post: list[int] | tuple[int, ...],
        n_layers_time: int,
        ch_init_time: int,
        kernel_size: list[int] | tuple[int, ...],
        mode: str,
        batch_norm: bool = False,
        downsamp_fac: int = 1,
        stride: int = 1,
        padding_mode: str = "zeros",
        device: str = "cpu",
    ) -> None:
        del data_shape, downsamp_fac, padding_mode

        torch = _require_torch()
        super().__init__()
        if mode != "2D":
            raise NotImplementedError("Only the paper's 2D MIMO CNN checkpoints are supported.")
        if batch_norm:
            raise NotImplementedError("Batch-normalized CNN checkpoints are not supported yet.")

        self.n_layers_pre = n_layers_pre
        self.n_layers_post = n_layers_post
        self.ch_layers_pre = tuple(ch_layers_pre)
        self.ch_layers_post = tuple(ch_layers_post)
        self.n_layers_time = n_layers_time
        self.ch_init_time = ch_init_time
        self.kernel_size = tuple(kernel_size)
        self.mode = mode
        self.device = torch.device(device)
        self.dim_time = self.ch_layers_pre[-1]

        if n_layers_time == 0:
            ch_time = (2 * self.dim_time,)
        elif n_layers_time == 1:
            ch_time = (ch_init_time, 2 * self.dim_time)
        elif n_layers_time == 2:
            ch_time = (ch_init_time, self.dim_time, 2 * self.dim_time)
        elif n_layers_time == 3:
            ch_time = (ch_init_time, self.dim_time, self.dim_time, 2 * self.dim_time)
        else:
            raise NotImplementedError(f"Unsupported time-MLP depth: {n_layers_time}")

        self.time_mlp = torch.nn.Sequential()
        for i in range(n_layers_time):
            self.time_mlp.add_module(f"time_linear{i + 1}", torch.nn.Linear(ch_time[i], ch_time[i + 1]))
            if i < n_layers_time - 1:
                self.time_mlp.add_module(f"act_time{i + 1}", torch.nn.ReLU())

        self.cnn_pre = torch.nn.Sequential()
        for i in range(n_layers_pre):
            self.cnn_pre.add_module(
                f"conv_pre{i}",
                torch.nn.Conv2d(
                    self.ch_layers_pre[i],
                    self.ch_layers_pre[i + 1],
                    stride=stride,
                    kernel_size=self.kernel_size,
                    padding="same",
                ),
            )
            if i < n_layers_pre - 1:
                self.cnn_pre.add_module(f"act_pre{i + 1}", torch.nn.ReLU())

        self.cnn_post = torch.nn.Sequential()
        for i in range(n_layers_post):
            self.cnn_post.add_module(
                f"conv_post{i}",
                torch.nn.Conv2d(
                    self.ch_layers_post[i],
                    self.ch_layers_post[i + 1],
                    stride=stride,
                    kernel_size=self.kernel_size,
                    padding="same",
                ),
            )
            if i < n_layers_post - 1:
                self.cnn_post.add_module(f"act_post{i + 1}", torch.nn.ReLU())

        self.to(self.device)

    @staticmethod
    def _positional_embedding(t: Any, dim: int) -> Any:
        torch = _require_torch()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=t.device))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if dim % 2 != 0:
            emb = torch.nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb

    def forward(self, x: Any, t: Any) -> Any:
        t_emb = self.time_mlp(self._positional_embedding(t, self.ch_init_time))
        scale = t_emb[:, : self.dim_time]
        shift = t_emb[:, self.dim_time :]

        x = self.cnn_pre(x)
        x = x + scale[:, :, None, None] * x + shift[:, :, None, None]
        return self.cnn_post(x)


class FeslPretrainedDenoiser:
    """Pretrained angular-domain epsilon predictor from Fesl et al.

    The checkpoint expects complex channels represented as two real channels
    with shape ``[batch, 2, n_rx, n_tx]``. Public methods accept and return
    complex NumPy arrays in angular-domain shape ``[n_rx, n_tx]`` or
    ``[batch, n_rx, n_tx]``.
    """

    def __init__(
        self,
        model: Any,
        *,
        alpha_bar: np.ndarray,
        betas: np.ndarray,
        device: str = "auto",
    ) -> None:
        torch = _require_torch()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.alpha_bar = np.asarray(alpha_bar, dtype=np.float64)
        self.betas = np.asarray(betas, dtype=np.float64)

    @property
    def num_timesteps(self) -> int:
        return int(self.alpha_bar.shape[0])

    @property
    def snr(self) -> np.ndarray:
        return self.alpha_bar / (1.0 - self.alpha_bar)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> "FeslPretrainedDenoiser":
        """Load a pretrained model from an official ``best_models_dm_paper`` subdir."""
        torch = _require_torch()
        model_dir = Path(model_dir)
        params_path = model_dir / "sim_params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing simulation parameters: {params_path}")

        with params_path.open("r", encoding="utf-8") as f:
            params = json.load(f)

        cnn_dict = dict(params["unet_dict"])
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        cnn_dict["device"] = resolved_device
        model = _FeslCNN(**cnn_dict)

        if checkpoint_path is None:
            train_models = sorted((model_dir / "train_models").glob("*.pt"))
            if not train_models:
                raise FileNotFoundError(f"No .pt checkpoints found in {model_dir / 'train_models'}")
            checkpoint_path = train_models[-1]

        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        state = checkpoint["model"]
        model_state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        model.load_state_dict(model_state)

        alpha_bar = state["alphas_cumprod"].detach().cpu().numpy()
        betas = state["betas"].detach().cpu().numpy()
        return cls(model, alpha_bar=alpha_bar, betas=betas, device=resolved_device)

    def predict_noise(self, H_tilde_t: np.ndarray, timestep: int) -> np.ndarray:
        """Predict ``epsilon_theta(H_tilde_t, t)`` for a zero-based DM timestep."""
        torch = _require_torch()
        original = np.asarray(H_tilde_t)
        if original.ndim == 2:
            batched = original[None, ...]
            squeeze = True
        elif original.ndim == 3:
            batched = original
            squeeze = False
        else:
            raise ValueError("H_tilde_t must have shape [n_rx, n_tx] or [batch, n_rx, n_tx].")
        if not np.iscomplexobj(batched):
            raise TypeError("H_tilde_t must be a complex-valued array.")
        if not 0 <= timestep < self.num_timesteps:
            raise ValueError(f"timestep must be in [0, {self.num_timesteps - 1}], got {timestep}.")

        real_channels = np.stack((batched.real, batched.imag), axis=1).astype(np.float32, copy=False)
        x = torch.from_numpy(real_channels).to(self.device)
        t = torch.full((batched.shape[0],), int(timestep), dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = self.model(x, t).detach().cpu().numpy()

        complex_pred = pred[:, 0, ...] + 1j * pred[:, 1, ...]
        if squeeze:
            return complex_pred[0]
        return complex_pred


def load_fesl_pretrained_denoiser(
    model_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> FeslPretrainedDenoiser:
    """Convenience loader for the official pretrained MIMO diffusion checkpoints."""
    return FeslPretrainedDenoiser.from_model_dir(
        model_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )
