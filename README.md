# Mass Mapping Starter

Minimal, shareable weak-lensing mass-mapping starter repository.

## Repository layout

- `scripts/` — runnable command-line scripts
  - `download_kappatng_runs.py`
  - `prepare_kappatng_subset.py`
  - `generate_shear_from_kappa.py`
  - `plot_bundle_outputs.py`
  - `prepare_noise_mask_sizes.py`
- `operators.py`, `noise_mask.py` — core NumPy operators and noise/mask helpers
- `data/` — reference assets and example outputs
- `notebooks/simple_forward_model_demo.ipynb` — short end-to-end notebook demo

## Included example outputs

- `data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz`  
  `kappa.shape = (10, 384, 384)`
- `data/shear_subset_lp001_runs001-010_wlmmuq_384.npz`
- `data/kappa_subset_lp001_runs001-010_wlmmuq_256.npz`
- `data/shear_subset_lp001_runs001-010_wlmmuq_256.npz`
- `data/cosmos_noise_mask_384.npz`
- `data/cosmos_noise_mask_256.npz`
- Plot set in `data/plots/`

## Requirements

- Python 3.10+
- `numpy`
- `h5py`
- `matplotlib` (for plotting)

## Quick start (384x384)

Run from repository root:

```bash
# 1) Download source runs (example: 10 runs from LP001)
python scripts/download_kappatng_runs.py \
  --dataset fullphys \
  --lp-index 1 \
  --start-run 1 \
  --n-runs 10 \
  --output-root data/kappaTNG_fullphys

# 2) Build kappa subset (wlmmuq-like redshift combination: z01..z40)
python scripts/prepare_kappatng_subset.py \
  --ktng-dir data/kappaTNG_fullphys \
  --lp-index 1 \
  --start-run 1 \
  --n-maps 10 \
  --redshift-mode wlmmuq \
  --crop-size 384 \
  --remove-mean \
  -o data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz

# 3) Generate noisy/masked shear maps
python scripts/generate_shear_from_kappa.py \
  --kappa-file data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz \
  --seed 42 \
  -o data/shear_subset_lp001_runs001-010_wlmmuq_384.npz

# 4) Plot diagnostics
python scripts/plot_bundle_outputs.py \
  --kappa-file data/kappa_subset_lp001_runs001-010_wlmmuq_384.npz \
  --shear-file data/shear_subset_lp001_runs001-010_wlmmuq_384.npz \
  --output-dir data/plots \
  --max-maps 10
```

To generate more than 10 maps, increase both `--n-runs` and `--n-maps`.

## 256x256 workflow

```bash
# Ensure 256 mask/noise asset exists (already included in this repo)
python scripts/prepare_noise_mask_sizes.py \
  --input-file data/cosmos_noise_mask_384.npz \
  --sizes 256 \
  --output-dir data

# Build 256 subset
python scripts/prepare_kappatng_subset.py \
  --ktng-dir data/kappaTNG_fullphys \
  --lp-index 1 \
  --start-run 1 \
  --n-maps 10 \
  --redshift-mode wlmmuq \
  --crop-size 256 \
  --remove-mean \
  -o data/kappa_subset_lp001_runs001-010_wlmmuq_256.npz

# Generate 256 shear subset
python scripts/generate_shear_from_kappa.py \
  --kappa-file data/kappa_subset_lp001_runs001-010_wlmmuq_256.npz \
  --seed 42 \
  -o data/shear_subset_lp001_runs001-010_wlmmuq_256.npz
```

## Noise model

Noise is added as complex Gaussian per pixel:

\[
n(x,y) = \left[\mathcal{N}(0,1) + i\,\mathcal{N}(0,1)\right]\sigma_{\mathrm{noise}}(x,y).
\]

`std_noise` is spatially varying and derived from COSMOS galaxy catalog statistics. In the original COSMOS construction:

\[
\sigma_{\mathrm{noise}}(x,y)=\sigma_e \frac{\sqrt{\sum_i w_i^2}}{\sum_i w_i},
\]

with sums over galaxies in each pixel. This is effectively galaxy-density / weight dependent.

## Output format (`generate_shear_from_kappa.py`)

- `kappa_true` — float32, `(n_maps, nx, ny)`
- `gamma_clean` — complex64, `(n_maps, nx, ny)`
- `gamma_noisy` — complex64, `(n_maps, nx, ny)`
- `noise` — complex64, `(n_maps, nx, ny)`
- `kappa_ks_e`, `kappa_ks_b` — float32, `(n_maps, nx, ny)`
- `std_noise` — float32, `(nx, ny)`
- `mask` — bool, `(nx, ny)`
- `extent` — optional float32, `(4,)`

## Notebook demo

Open:

`notebooks/simple_forward_model_demo.ipynb`

It demonstrates loading kappa maps, converting to shear, adding COSMOS mask/noise, and plotting results.
