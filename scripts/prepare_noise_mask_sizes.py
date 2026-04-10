#!/usr/bin/env python3
"""Create center-cropped COSMOS noise/mask files at target square sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _center_crop(arr: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    nx, ny = arr.shape[-2:]
    if nx != ny:
        raise ValueError(f"Expected square map, got {arr.shape}.")
    if size > nx:
        raise ValueError(f"Requested size={size} is larger than input size={nx}.")
    if size <= 0:
        raise ValueError("Requested size must be > 0.")
    bx = (nx - size) // 2
    by = (ny - size) // 2
    return arr[..., bx : bx + size, by : by + size], bx, by


def _crop_extent(
    extent: np.ndarray | None,
    nx: int,
    ny: int,
    bx: int,
    by: int,
    size: int,
) -> np.ndarray:
    if extent is None or extent.size != 4:
        return np.array([], dtype=np.float32)
    x_min, x_max, y_min, y_max = [float(v) for v in extent]
    dx = (x_max - x_min) / ny
    dy = (y_max - y_min) / nx
    cropped = np.array(
        [
            x_min + by * dx,
            x_min + (by + size) * dx,
            y_min + bx * dy,
            y_min + (bx + size) * dy,
        ],
        dtype=np.float32,
    )
    return cropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "cosmos_noise_mask_384.npz",
        help="Input reference .npz (must include std_noise and mask).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Target output sizes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Where output files are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files.",
    )
    args = parser.parse_args()

    payload = np.load(args.input_file)
    std_noise = np.asarray(payload["std_noise"], dtype=np.float32)
    mask = np.asarray(payload["mask"], dtype=bool)
    extent = np.asarray(payload["extent"], dtype=np.float32) if "extent" in payload.files else None

    nx, ny = std_noise.shape
    if mask.shape != (nx, ny):
        raise ValueError(
            f"Shape mismatch between std_noise {std_noise.shape} and mask {mask.shape}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        std_crop, bx, by = _center_crop(std_noise, size)
        mask_crop, _, _ = _center_crop(mask, size)
        ext_crop = _crop_extent(extent, nx, ny, bx, by, size)
        out_file = args.output_dir / f"cosmos_noise_mask_{size}.npz"
        if out_file.exists() and not args.overwrite:
            print(f"Skip existing file: {out_file}")
            continue
        np.savez_compressed(
            out_file,
            std_noise=std_crop.astype(np.float32),
            mask=mask_crop.astype(bool),
            extent=ext_crop,
        )
        print(f"Saved {out_file} with shape {std_crop.shape}")


if __name__ == "__main__":
    main()
