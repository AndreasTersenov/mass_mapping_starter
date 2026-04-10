"""Mask/noise helpers for NumPy shear simulations."""

from __future__ import annotations

import numpy as np


def load_noise_mask(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load reference noise and mask arrays from a .npz file.

    Expected keys:
    - std_noise: float array, shape (nx, ny)
    - mask: boolean array, shape (nx, ny)
    Optional key:
    - extent: float array, shape (4,)
    """
    payload = np.load(path)
    std_noise = np.asarray(payload["std_noise"], dtype=np.float32)
    mask = np.asarray(payload["mask"], dtype=bool)
    extent = np.asarray(payload["extent"]) if "extent" in payload.files else None
    return std_noise, mask, extent


def apply_mask_and_noise(
    gamma_clean: np.ndarray,
    std_noise: np.ndarray,
    mask: np.ndarray,
    seed: int | None = None,
    inpainting: bool = False,
    return_noise: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply survey mask and complex Gaussian noise to clean shear maps.

    Parameters
    ----------
    gamma_clean
        Complex shear maps with shape (..., nx, ny).
    std_noise
        Per-pixel noise std map with shape (nx, ny).
    mask
        Survey mask with shape (nx, ny), True where observations exist.
    seed
        Optional random seed for reproducibility.
    inpainting
        If False, masked pixels are set to 0 in the noisy map.
        If True, masked pixels are filled with random noise.
    """
    gamma_clean = np.asarray(gamma_clean)
    if not np.iscomplexobj(gamma_clean):
        raise ValueError("`gamma_clean` must be complex-valued.")

    std_noise = np.asarray(std_noise, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    if gamma_clean.shape[-2:] != std_noise.shape:
        raise ValueError(
            f"Shape mismatch: gamma has {gamma_clean.shape[-2:]}, std_noise has {std_noise.shape}."
        )
    if mask.shape != std_noise.shape:
        raise ValueError(f"Shape mismatch: mask has {mask.shape}, expected {std_noise.shape}.")

    gamma_masked = gamma_clean.copy()
    gamma_masked[..., ~mask] = 0.0

    rng = np.random.default_rng(seed)
    noise_real = rng.standard_normal(gamma_clean.shape, dtype=np.float32)
    noise_imag = rng.standard_normal(gamma_clean.shape, dtype=np.float32)
    noise = (noise_real + 1j * noise_imag) * std_noise

    if not inpainting:
        noise[..., ~mask] = 0.0

    gamma_noisy = gamma_masked + noise
    if return_noise:
        return gamma_noisy, noise
    return gamma_noisy
