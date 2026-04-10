#!/usr/bin/env python3
"""Create quick-look plots for kappa and shear bundle outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _percentile_limits(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    lo = float(np.nanpercentile(arr, low))
    hi = float(np.nanpercentile(arr, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if lo == hi:
            hi = lo + 1e-6
    return lo, hi


def _save_kappa_plots(
    kappa: np.ndarray,
    run_ids: np.ndarray | None,
    out_dir: Path,
    max_maps: int,
) -> None:
    n = min(max_maps, kappa.shape[0])
    if n == 0:
        return

    vmin, vmax = _percentile_limits(kappa[:n], low=1.0, high=99.0)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        title = f"kappa map {i}"
        if run_ids is not None and i < len(run_ids):
            title += f" (run {int(run_ids[i]):03d})"
        im = ax.imshow(kappa[i], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes_flat.tolist(), shrink=0.85, label="kappa")
    fig.savefig(out_dir / "kappa_subset_overview.png", dpi=160)
    plt.close(fig)

    for i in range(n):
        fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
        title = f"kappa[{i}]"
        if run_ids is not None and i < len(run_ids):
            title += f" run {int(run_ids[i]):03d}"
        im = ax.imshow(kappa[i], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(out_dir / f"kappa_map_{i:02d}.png", dpi=160)
        plt.close(fig)


def _save_noise_mask_plot(mask: np.ndarray, std_noise: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(mask.astype(float), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("mask (1=valid, 0=masked)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    vmin_sn, vmax_sn = _percentile_limits(std_noise, low=0.0, high=99.5)
    im1 = axes[1].imshow(std_noise, cmap="viridis", vmin=vmin_sn, vmax=vmax_sn)
    axes[1].set_title("std_noise")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    fig.savefig(out_dir / "mask_and_std_noise.png", dpi=160)
    plt.close(fig)


def _save_shear_component_plots(payload: dict[str, np.ndarray], out_dir: Path, max_maps: int) -> None:
    kappa = payload["kappa_true"]
    gamma_clean = payload["gamma_clean"]
    gamma_noisy = payload["gamma_noisy"]
    noise = payload["noise"]
    kappa_ks_e = payload["kappa_ks_e"]
    kappa_ks_b = payload["kappa_ks_b"]

    n = min(max_maps, kappa.shape[0])
    for i in range(n):
        g1_clean = np.real(gamma_clean[i])
        g2_clean = np.imag(gamma_clean[i])
        g1_noisy = np.real(gamma_noisy[i])
        g2_noisy = np.imag(gamma_noisy[i])
        n1 = np.real(noise[i])
        n2 = np.imag(noise[i])

        kvmin, kvmax = _percentile_limits(np.stack([kappa[i], kappa_ks_e[i], kappa_ks_b[i]], axis=0))
        gvmin, gvmax = _percentile_limits(np.stack([g1_clean, g2_clean, g1_noisy, g2_noisy], axis=0))
        nvmin, nvmax = _percentile_limits(np.stack([n1, n2], axis=0))

        panels = [
            (kappa[i], "kappa_true", kvmin, kvmax),
            (g1_clean, "gamma1_clean", gvmin, gvmax),
            (g2_clean, "gamma2_clean", gvmin, gvmax),
            (g1_noisy, "gamma1_noisy", gvmin, gvmax),
            (g2_noisy, "gamma2_noisy", gvmin, gvmax),
            (n1, "noise_gamma1", nvmin, nvmax),
            (n2, "noise_gamma2", nvmin, nvmax),
            (kappa_ks_e[i], "kappa_ks_e", kvmin, kvmax),
            (kappa_ks_b[i], "kappa_ks_b", kvmin, kvmax),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        for ax, (arr, title, vmin, vmax) in zip(axes.flat, panels):
            im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"shear pipeline diagnostics: map {i}", fontsize=12)
        fig.savefig(out_dir / f"shear_diagnostics_map_{i:02d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappa-file", type=Path, required=True, help="Input kappa subset (.npz).")
    parser.add_argument("--shear-file", type=Path, required=True, help="Input shear subset (.npz).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("collab_massmap_bundle/data/plots"),
        help="Directory where PNG files are written.",
    )
    parser.add_argument("--max-maps", type=int, default=10, help="Maximum number of maps to plot.")
    args = parser.parse_args()

    kappa_payload = np.load(args.kappa_file)
    shear_payload = np.load(args.shear_file)

    kappa = np.asarray(kappa_payload["kappa"], dtype=np.float32)
    run_ids = np.asarray(kappa_payload["run_ids"]) if "run_ids" in kappa_payload.files else None

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_kappa_plots(kappa, run_ids, out_dir, max_maps=args.max_maps)
    _save_noise_mask_plot(
        mask=np.asarray(shear_payload["mask"], dtype=bool),
        std_noise=np.asarray(shear_payload["std_noise"], dtype=np.float32),
        out_dir=out_dir,
    )
    _save_shear_component_plots(
        payload={
            "kappa_true": np.asarray(shear_payload["kappa_true"], dtype=np.float32),
            "gamma_clean": np.asarray(shear_payload["gamma_clean"], dtype=np.complex64),
            "gamma_noisy": np.asarray(shear_payload["gamma_noisy"], dtype=np.complex64),
            "noise": np.asarray(shear_payload["noise"], dtype=np.complex64),
            "kappa_ks_e": np.asarray(shear_payload["kappa_ks_e"], dtype=np.float32),
            "kappa_ks_b": np.asarray(shear_payload["kappa_ks_b"], dtype=np.float32),
        },
        out_dir=out_dir,
        max_maps=args.max_maps,
    )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
