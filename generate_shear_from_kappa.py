#!/usr/bin/env python3
"""Generate clean/noisy shear maps from a kappa subset (.npz)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from noise_mask import apply_mask_and_noise, load_noise_mask
from operators import kappa_to_shear, shear_to_kappa


def _resolve_noise_mask_path(noise_mask_file: Path | None, map_size: int) -> Path:
    if noise_mask_file is not None:
        return noise_mask_file
    candidate = (
        Path(__file__).resolve().parent / "data" / f"cosmos_noise_mask_{map_size}.npz"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            "No default noise-mask file found for map size "
            f"{map_size} (expected {candidate}). "
            "Provide --noise-mask-file explicitly."
        )
    return candidate


def _load_kappa(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "kappa" not in payload.files:
        raise KeyError(f"{path} does not contain key 'kappa'.")
    kappa = np.asarray(payload["kappa"], dtype=np.float32)
    if kappa.ndim == 2:
        kappa = kappa[None, ...]
    if kappa.ndim != 3:
        raise ValueError(f"`kappa` must have shape (n_maps, nx, ny), got {kappa.shape}.")
    return kappa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kappa-file",
        type=Path,
        required=True,
        help="Input .npz produced by prepare_kappatng_subset.py.",
    )
    parser.add_argument(
        "--noise-mask-file",
        type=Path,
        default=None,
        help=(
            "Reference .npz with keys: std_noise, mask (and optional extent). "
            "If omitted, auto-selects data/cosmos_noise_mask_<map_size>.npz."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise.",
    )
    parser.add_argument(
        "--inpainting",
        action="store_true",
        help="If set, masked pixels are filled with random noise instead of zero.",
    )
    parser.add_argument(
        "--complex-conjugate",
        action="store_true",
        help="Apply the same optional orientation convention used in the main codebase.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/shear_subset_lp001.npz"),
        help="Output .npz file.",
    )
    args = parser.parse_args()

    kappa = _load_kappa(args.kappa_file)
    if kappa.shape[-1] != kappa.shape[-2]:
        raise ValueError(f"Expected square maps, got {kappa.shape[-2:]}.")
    noise_mask_file = _resolve_noise_mask_path(args.noise_mask_file, kappa.shape[-1])
    std_noise, mask, extent = load_noise_mask(str(noise_mask_file))

    if kappa.shape[-2:] != std_noise.shape:
        raise ValueError(
            f"Shape mismatch: kappa has {kappa.shape[-2:]}, "
            f"noise/mask has {std_noise.shape}. "
            "Use matching crop size in prepare_kappatng_subset.py."
        )

    gamma_clean = kappa_to_shear(
        kappa,
        mask=None,
        complex_conjugate=args.complex_conjugate,
        return_complex=True,
    )
    gamma_noisy, noise = apply_mask_and_noise(
        gamma_clean,
        std_noise=std_noise,
        mask=mask,
        seed=args.seed,
        inpainting=args.inpainting,
        return_noise=True,
    )
    kappa_ks_e, kappa_ks_b = shear_to_kappa(
        gamma_noisy,
        mask=mask,
        complex_conjugate=args.complex_conjugate,
        return_complex=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        kappa_true=kappa.astype(np.float32),
        gamma_clean=gamma_clean.astype(np.complex64),
        gamma_noisy=gamma_noisy.astype(np.complex64),
        noise=noise.astype(np.complex64),
        kappa_ks_e=kappa_ks_e.astype(np.float32),
        kappa_ks_b=kappa_ks_b.astype(np.float32),
        std_noise=std_noise.astype(np.float32),
        mask=mask.astype(bool),
        extent=(extent.astype(np.float32) if extent is not None else np.array([], dtype=np.float32)),
        seed=np.int32(args.seed),
        inpainting=np.bool_(args.inpainting),
    )

    print(f"Saved shear dataset to {args.output}")
    print(f"kappa/gamma shape: {kappa.shape}")
    print(f"noise-mask file: {noise_mask_file}")


if __name__ == "__main__":
    main()
