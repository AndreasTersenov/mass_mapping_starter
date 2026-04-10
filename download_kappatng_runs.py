#!/usr/bin/env python3
"""Download selected kappaTNG HDF5 run files into a local folder."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://idark.ipmu.jp/~jia.liu/data/kappaTNG"


def _build_url(dataset: str, lp_index: str, run_id: int) -> str:
    run = f"{run_id:03d}"
    return (
        f"{BASE_URL}/{dataset}/maps/LP{lp_index}/LP{lp_index}_run{run}_maps.hdf5"
    )


def _destination_file(output_root: Path, lp_index: str, run_id: int) -> Path:
    run = f"{run_id:03d}"
    return output_root / f"LP{lp_index}" / f"LP{lp_index}_run{run}_maps.hdf5"


def _download(url: str, destination: Path, overwrite: bool, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"Skip existing file: {destination}")
        return

    tmp_file = destination.with_suffix(destination.suffix + ".part")
    if tmp_file.exists():
        tmp_file.unlink()

    print(f"Downloading:\n  {url}\n  -> {destination}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total is not None else 0
            written = 0
            with open(tmp_file, "wb") as fout:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
                    written += len(chunk)
                    if total_bytes > 0:
                        progress = 100.0 * written / total_bytes
                        print(
                            f"  {written / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB "
                            f"({progress:.1f}%)",
                            end="\r",
                            flush=True,
                        )
            if total_bytes > 0:
                print(" " * 80, end="\r", flush=True)
    except urllib.error.HTTPError as err:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError(f"HTTP error for {url}: {err.code}") from err
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    shutil.move(tmp_file, destination)
    print(f"Done: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("fullphys", "dmonly"),
        default="fullphys",
        help="kappaTNG dataset family to download.",
    )
    parser.add_argument(
        "--lp-index",
        type=int,
        default=1,
        help="Lensing potential index (e.g., 1 for LP001).",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help="First run index (1-based).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of consecutive run files to download.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory where LP folders are created. "
            "Defaults to collab_massmap_bundle/data/kappaTNG_<dataset>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    if args.n_runs <= 0:
        raise ValueError("--n-runs must be > 0.")
    if args.start_run <= 0:
        raise ValueError("--start-run must be > 0.")

    lp_index = f"{args.lp_index:03d}"
    if args.output_root is None:
        output_root = Path("collab_massmap_bundle/data") / f"kappaTNG_{args.dataset}"
    else:
        output_root = args.output_root

    run_ids = range(args.start_run, args.start_run + args.n_runs)
    for run_id in run_ids:
        url = _build_url(args.dataset, lp_index, run_id)
        destination = _destination_file(output_root, lp_index, run_id)
        _download(
            url=url,
            destination=destination,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )

    print(
        "Finished download batch:\n"
        f"  dataset={args.dataset}, LP{lp_index}, runs={args.start_run:03d}-"
        f"{args.start_run + args.n_runs - 1:03d}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
