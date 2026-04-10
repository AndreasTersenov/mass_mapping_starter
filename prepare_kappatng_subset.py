#!/usr/bin/env python3
"""Export a tiny subset of kappaTNG convergence maps to a portable .npz file.

This script is intentionally simple (NumPy + h5py only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _center_crop(arr: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None:
        return arr
    nx, ny = arr.shape[-2:]
    if crop_size > nx or crop_size > ny:
        raise ValueError(
            f"crop_size={crop_size} is larger than map size ({nx}, {ny})."
        )
    bx = (nx - crop_size) // 2
    by = (ny - crop_size) // 2
    return arr[..., bx : bx + crop_size, by : by + crop_size]


def _list_redshift_groups(h5f: h5py.File) -> list[str]:
    groups = [key for key in sorted(h5f.keys()) if key.lower().startswith("z")]
    if not groups:
        raise RuntimeError("Could not find redshift groups ('zXX') in the run file.")
    return groups


def _combine_redshift_stack(
    stack: np.ndarray,
    redshift_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    nz = stack.shape[0]
    if redshift_weights is None:
        weights = np.full(nz, 1.0 / nz, dtype=np.float64)
    else:
        weights = np.asarray(redshift_weights, dtype=np.float64).reshape(-1)
        if len(weights) != nz:
            raise ValueError(
                f"weights-file has {len(weights)} values, but selected stack has {nz} planes."
            )
        if np.any(weights < 0):
            raise ValueError("weights-file must contain non-negative values.")
        if not np.any(weights > 0):
            raise ValueError("weights-file cannot be all zeros.")
        weights = weights / np.sum(weights)
    kappa = np.einsum("z,zxy->xy", weights, stack, optimize=True)
    return kappa, weights


def _load_map_from_run(
    run_file: Path,
    redshift_mode: str,
    redshift_index: int,
    redshift_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with h5py.File(run_file, "r", swmr=True) as f:
        groups = _list_redshift_groups(f)
        if redshift_mode == "single":
            if redshift_index < 0 or redshift_index >= len(groups):
                raise ValueError(
                    f"redshift_index={redshift_index} out of range for {len(groups)} groups."
                )
            groups_used = [groups[redshift_index]]
            kappa = f[f"{groups_used[0]}/kappa"][:]
            weights_used = None
        elif redshift_mode == "average":
            groups_used = groups
            stack = [f[f"{group}/kappa"][:] for group in groups_used]
            stack = np.stack(stack, axis=0)
            kappa, weights_used = _combine_redshift_stack(stack, redshift_weights)
        elif redshift_mode == "wlmmuq":
            # Match wlmmuq behavior: use sorted(file.keys())[1:], which corresponds
            # to z01..z40 for current kappaTNG HDF5 files.
            if len(groups) < 2:
                raise ValueError(
                    "wlmmuq mode needs at least two redshift groups to skip the first one."
                )
            groups_used = groups[1:]
            stack = [f[f"{group}/kappa"][:] for group in groups_used]
            stack = np.stack(stack, axis=0)
            kappa, weights_used = _combine_redshift_stack(stack, redshift_weights)
        else:
            raise ValueError(f"Unknown redshift_mode: {redshift_mode}")
    return np.asarray(kappa, dtype=np.float32), groups_used, weights_used


def _run_id_to_file(ktng_dir: Path, lp_index: str, run_id: int) -> Path:
    run = f"{run_id:03d}"
    return ktng_dir / f"LP{lp_index}" / f"LP{lp_index}_run{run}_maps.hdf5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ktng-dir",
        type=Path,
        required=True,
        help="Path to the kappaTNG root directory.",
    )
    parser.add_argument(
        "--lp-index",
        type=int,
        default=1,
        help="Lensing potential index (e.g., 1 for LP001).",
    )
    parser.add_argument(
        "--n-maps",
        type=int,
        default=5,
        help="Number of maps/runs to export.",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help="First run index (1-based).",
    )
    parser.add_argument(
        "--redshift-mode",
        choices=("single", "average", "wlmmuq"),
        default="wlmmuq",
        help=(
            "How to collapse redshift planes into one kappa map per run. "
            "'wlmmuq' mimics wlmmuq (use z01..z40, weighted average if provided)."
        ),
    )
    parser.add_argument(
        "--redshift-index",
        type=int,
        default=0,
        help="Used when --redshift-mode single.",
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one weight per redshift plane. "
            "Used only with --redshift-mode average."
        ),
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=384,
        help="Center crop size. Use 0 to disable cropping.",
    )
    parser.add_argument(
        "--remove-mean",
        action="store_true",
        help="Subtract per-map mean after loading/cropping.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/kappa_subset_lp001.npz"),
        help="Output .npz file.",
    )
    args = parser.parse_args()

    lp_index = f"{args.lp_index:03d}"
    crop_size = None if args.crop_size == 0 else args.crop_size
    run_ids = list(range(args.start_run, args.start_run + args.n_maps))
    if args.redshift_mode == "single" and args.weights_file is not None:
        raise ValueError("--weights-file is not used with --redshift-mode single.")
    if args.weights_file is not None:
        redshift_weights = np.loadtxt(args.weights_file, dtype=np.float64)
        redshift_weights = np.atleast_1d(redshift_weights).reshape(-1)
    else:
        redshift_weights = None

    kappas = []
    redshift_groups_used: list[str] | None = None
    redshift_weights_used: np.ndarray | None = None
    for run_id in run_ids:
        run_file = _run_id_to_file(args.ktng_dir, lp_index, run_id)
        if not run_file.exists():
            raise FileNotFoundError(f"Missing kappaTNG run file: {run_file}")
        kappa, groups_used, weights_used = _load_map_from_run(
            run_file,
            redshift_mode=args.redshift_mode,
            redshift_index=args.redshift_index,
            redshift_weights=redshift_weights,
        )
        if redshift_groups_used is None:
            redshift_groups_used = groups_used
            redshift_weights_used = weights_used
        elif redshift_groups_used != groups_used:
            raise RuntimeError(
                f"Redshift groups differ across runs: {redshift_groups_used} vs {groups_used}"
            )
        kappa = _center_crop(kappa, crop_size)
        if args.remove_mean:
            kappa = kappa - np.mean(kappa, dtype=np.float64)
        kappas.append(kappa.astype(np.float32, copy=False))

    kappa_subset = np.stack(kappas, axis=0)
    if redshift_groups_used is None:
        raise RuntimeError("No kappa maps were loaded.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        kappa=kappa_subset,
        run_ids=np.asarray(run_ids, dtype=np.int32),
        lp_index=lp_index,
        redshift_mode=args.redshift_mode,
        redshift_index=np.int32(args.redshift_index),
        redshift_groups=np.asarray(redshift_groups_used),
        redshift_weights=(
            np.asarray([], dtype=np.float32)
            if redshift_weights_used is None
            else redshift_weights_used.astype(np.float32)
        ),
        crop_size=np.int32(0 if crop_size is None else crop_size),
    )

    print(f"Saved {kappa_subset.shape[0]} maps to {args.output}")
    print(f"Map shape: {kappa_subset.shape[1:]}")


if __name__ == "__main__":
    main()
