"""Minimal NumPy weak-lensing operators for collaborator handoff.

This file intentionally avoids framework dependencies (no PyTorch).
"""

from __future__ import annotations

import numpy as np


def _fft_grids(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Fourier frequency grids with shape (nx, ny)."""
    kx, ky = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="xy")
    return kx, ky


def _check_pair(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {name_a} has shape {a.shape}, {name_b} has shape {b.shape}."
        )


def _check_mask(mask: np.ndarray, shape: tuple[int, ...]) -> None:
    if mask.dtype != bool:
        raise ValueError("`mask` must be a boolean array.")
    if mask.shape != shape[-2:]:
        raise ValueError(
            f"Mask shape mismatch: expected {shape[-2:]}, got {mask.shape}."
        )


def ks93(g1: np.ndarray, g2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaiser-Squires operator: shear -> convergence (E/B).

    Parameters
    ----------
    g1, g2
        Shear components with shape (..., nx, ny).
    """
    g1 = np.asarray(g1)
    g2 = np.asarray(g2)
    _check_pair(g1, g2, "g1", "g2")

    nx, ny = g1.shape[-2:]
    kx, ky = _fft_grids(nx, ny)

    g1_hat = np.fft.fft2(g1, axes=(-2, -1))
    g2_hat = np.fft.fft2(g2, axes=(-2, -1))

    p1 = kx * kx - ky * ky
    p2 = 2.0 * kx * ky
    k2 = kx * kx + ky * ky
    k2 = k2.copy()
    k2[0, 0] = 1.0

    k_e_hat = (p1 * g1_hat + p2 * g2_hat) / k2
    k_b_hat = -(p2 * g1_hat - p1 * g2_hat) / k2

    k_e = np.fft.ifft2(k_e_hat, axes=(-2, -1)).real
    k_b = np.fft.ifft2(k_b_hat, axes=(-2, -1)).real
    return k_e, k_b


def ks93inv(k_e: np.ndarray, k_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse Kaiser-Squires operator: convergence (E/B) -> shear.

    Parameters
    ----------
    k_e, k_b
        Convergence components with shape (..., nx, ny).
    """
    k_e = np.asarray(k_e)
    k_b = np.asarray(k_b)
    _check_pair(k_e, k_b, "k_e", "k_b")

    nx, ny = k_e.shape[-2:]
    kx, ky = _fft_grids(nx, ny)

    k_e_hat = np.fft.fft2(k_e, axes=(-2, -1))
    k_b_hat = np.fft.fft2(k_b, axes=(-2, -1))

    p1 = kx * kx - ky * ky
    p2 = 2.0 * kx * ky
    k2 = kx * kx + ky * ky
    k2 = k2.copy()
    k2[0, 0] = 1.0

    g1_hat = (p1 * k_e_hat - p2 * k_b_hat) / k2
    g2_hat = (p2 * k_e_hat + p1 * k_b_hat) / k2

    g1 = np.fft.ifft2(g1_hat, axes=(-2, -1)).real
    g2 = np.fft.ifft2(g2_hat, axes=(-2, -1)).real
    return g1, g2


def kappa_to_shear(
    kappa_e: np.ndarray,
    kappa_b: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    complex_conjugate: bool = False,
    return_complex: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert convergence maps to shear maps.

    `kappa_e` has shape (..., nx, ny). If `kappa_b` is omitted, it is set to zero.
    """
    kappa_e = np.asarray(kappa_e)
    if kappa_b is None:
        kappa_b = np.zeros_like(kappa_e)
    else:
        kappa_b = np.asarray(kappa_b)
    _check_pair(kappa_e, kappa_b, "kappa_e", "kappa_b")

    # Match the convention used in the main project when requested.
    if complex_conjugate:
        kappa_b = -kappa_b

    g1, g2 = ks93inv(kappa_e, kappa_b)

    if complex_conjugate:
        g2 = -g2

    if mask is not None:
        mask = np.asarray(mask)
        _check_mask(mask, g1.shape)
        g1 = g1.copy()
        g2 = g2.copy()
        g1[..., ~mask] = 0.0
        g2[..., ~mask] = 0.0

    if return_complex:
        return g1 + 1j * g2
    return g1, g2


def shear_to_kappa(
    gamma1: np.ndarray,
    gamma2: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    complex_conjugate: bool = False,
    return_complex: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert shear maps to convergence maps (E/B).

    If `gamma2` is omitted, `gamma1` is interpreted as a complex shear map.
    """
    if gamma2 is None:
        gamma = np.asarray(gamma1)
        if not np.iscomplexobj(gamma):
            raise ValueError("If `gamma2` is None, `gamma1` must be a complex array.")
        g1 = gamma.real
        g2 = gamma.imag
    else:
        g1 = np.asarray(gamma1)
        g2 = np.asarray(gamma2)
        _check_pair(g1, g2, "gamma1", "gamma2")

    if mask is not None:
        mask = np.asarray(mask)
        _check_mask(mask, g1.shape)
        g1 = g1.copy()
        g2 = g2.copy()
        g1[..., ~mask] = 0.0
        g2[..., ~mask] = 0.0

    if complex_conjugate:
        g2 = -g2

    k_e, k_b = ks93(g1, g2)

    if complex_conjugate:
        k_b = -k_b

    if return_complex:
        return k_e + 1j * k_b
    return k_e, k_b
