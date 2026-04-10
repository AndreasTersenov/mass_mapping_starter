# Mass mapping starter pack

This folder lets collaborators:
1. download selected kappaTNG runs,
2. build kappa subsets with wlmmuq-style redshift combination,
3. generate realistic noisy/masked shear maps,
4. produce quick-look plots.

## Included examples

- `data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz`  
  10 combined-redshift kappa maps, shape `(10, 384, 384)`.
- `data/shear_subset_lp001_runs001-010_wlmmuq_384.npz`  
  Shear outputs for the same 10 maps.
- `data/cosmos_noise_mask_384.npz` and `data/cosmos_noise_mask_256.npz`  
  Precomputed COSMOS-derived noise/mask files.

So yes: the currently produced example maps are **384x384**.

## Scripts

- `download_kappatng_runs.py`  
  Download chosen LP/run HDF5 files from the official kappaTNG server.
- `prepare_kappatng_subset.py`  
  Export a compact kappa subset (`.npz`) from downloaded HDF5 files.
- `generate_shear_from_kappa.py`  
  Apply KS forward model and COSMOS-like mask/noise.
- `plot_bundle_outputs.py`  
  Save viridis quick-look PNGs (includes `gamma1` and `gamma2`).
- `prepare_noise_mask_sizes.py`  
  Generate additional center-cropped mask/noise assets (e.g. 256 from 384).
- `simple_forward_model_demo.ipynb`  
  Short notebook example using these scripts/modules end-to-end.
- `operators.py`, `noise_mask.py`  
  Core NumPy operators and noise helpers.

## Requirements

- Python 3.10+
- `numpy`
- `h5py`
- `matplotlib` (for plots)

## How to make 10 maps (or more)

From repository root:

```bash
# 1) Download kappaTNG runs (example: 10 runs from LP001)
python collab_massmap_bundle/download_kappatng_runs.py \
  --dataset fullphys \
  --lp-index 1 \
  --start-run 1 \
  --n-runs 10 \
  --output-root collab_massmap_bundle/data/kappaTNG_fullphys

# 2) Build kappa subset (wlmmuq-style redshift combination z01..z40)
python collab_massmap_bundle/prepare_kappatng_subset.py \
  --ktng-dir collab_massmap_bundle/data/kappaTNG_fullphys \
  --lp-index 1 \
  --start-run 1 \
  --n-maps 10 \
  --redshift-mode wlmmuq \
  --crop-size 384 \
  --remove-mean \
  -o collab_massmap_bundle/data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz

# 3) Generate shear maps (auto-selects cosmos_noise_mask_384.npz)
python collab_massmap_bundle/generate_shear_from_kappa.py \
  --kappa-file collab_massmap_bundle/data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz \
  --seed 42 \
  -o collab_massmap_bundle/data/shear_subset_lp001_runs001-010_wlmmuq_384.npz

# 4) Plot diagnostics (viridis colormap)
python collab_massmap_bundle/plot_bundle_outputs.py \
  --kappa-file collab_massmap_bundle/data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz \
  --shear-file collab_massmap_bundle/data/shear_subset_lp001_runs001-010_wlmmuq_384.npz \
  --output-dir collab_massmap_bundle/data/plots \
  --max-maps 10
```

To make **more than 10**, increase both:
- `download_kappatng_runs.py --n-runs`
- `prepare_kappatng_subset.py --n-maps`

## 256x256 option

This is supported directly:

```bash
# Build 256 mask/noise asset (already provided as data/cosmos_noise_mask_256.npz)
python collab_massmap_bundle/prepare_noise_mask_sizes.py \
  --input-file collab_massmap_bundle/data/cosmos_noise_mask_384.npz \
  --sizes 256 \
  --output-dir collab_massmap_bundle/data

# Build 256 kappa subset
python collab_massmap_bundle/prepare_kappatng_subset.py \
  --ktng-dir collab_massmap_bundle/data/kappaTNG_fullphys \
  --lp-index 1 \
  --start-run 1 \
  --n-maps 10 \
  --redshift-mode wlmmuq \
  --crop-size 256 \
  --remove-mean \
  -o collab_massmap_bundle/data/kappa_subset_lp001_runs001-010_wlmmuq_256.npz

# Generate 256 shear subset (auto-selects cosmos_noise_mask_256.npz)
python collab_massmap_bundle/generate_shear_from_kappa.py \
  --kappa-file collab_massmap_bundle/data/kappa_subset_lp001_runs001-010_wlmmuq_256.npz \
  --seed 42 \
  -o collab_massmap_bundle/data/shear_subset_lp001_runs001-010_wlmmuq_256.npz
```

## Noise model (what is added?)

- Noise is complex Gaussian per pixel:
  - `noise = (N(0,1) + i N(0,1)) * std_noise(x, y)`
- The `std_noise` map comes from COSMOS galaxy catalog statistics and is spatially varying.
- In the original COSMOS construction:
  \[
  \sigma_\mathrm{noise}(x,y) = \sigma_e \frac{\sqrt{\sum_i w_i^2}}{\sum_i w_i}
  \]
  where sums are over galaxies in that pixel.

So yes, it is effectively **galaxy-density / weight dependent** through the local per-pixel weights and counts.

## Output format

`generate_shear_from_kappa.py` writes:

- `kappa_true` (float32, `(n_maps, nx, ny)`)
- `gamma_clean` (complex64, `(n_maps, nx, ny)`)
- `gamma_noisy` (complex64, `(n_maps, nx, ny)`)
- `noise` (complex64, `(n_maps, nx, ny)`)
- `kappa_ks_e`, `kappa_ks_b` (float32, `(n_maps, nx, ny)`)
- `std_noise` (float32, `(nx, ny)`)
- `mask` (bool, `(nx, ny)`)
- `extent` (optional, `(4,)`)

## Notebook

Open and run:

`collab_massmap_bundle/simple_forward_model_demo.ipynb`

It loads a kappa subset, computes clean shear, adds COSMOS mask/noise, and plots all key outputs.
