# LGADNet: LAMOST DR12 CEMP Candidate Search

This repository contains the training, prediction, and screening pipeline used to search
for Carbon-Enhanced Metal-Poor (CEMP) candidates in LAMOST DR12, and the codes that
produce every figure in the accompanying paper.

- Four-parameter regression (`T_eff`, `log g`, `[Fe/H]`, `[C/Fe]`) from low-resolution
  LAMOST spectra.
- Systematic CEMP-candidate screening chain seeded by the predicted parameters.
- Final catalog of **4,568 high-confidence CEMP candidates** (median `[Fe/H] = -2.20`,
  median `[C/Fe] = 0.88`).

## Paper

*LAMOST DR12 Carbon-Enhanced Metal-Poor (CEMP) candidate search and catalog construction.*
The four-parameter predictions and the screening chain follow
Sects. 3-4 of the manuscript; Sect. 5 validates the sample via spectral statistics,
the GALAH external catalog, and the chemical-abundance / evolutionary-position
distribution.

## Data layout

Set the single root placeholder to your data tree and edit the "EDIT THIS root" line in
each script (or pass the equivalent `--` argument):

```python
LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root
```

Place the released files under this root as follows:

```
lgadnet_data/
  dr12_metadata.csv                      # LAMOST DR12 metadata (paths, magnitudes, ...)
  dr12_fits/                             # LAMOST DR12 spectra (FITS)
  apogee_dr17.csv                        # APOGEE DR17 catalog used for the metallicity cut
  galah_selected_data.csv                # GALAH catalog used for external validation
  lgadnet_dataset/                       # training datasets (produced by scripts/01_train)
  predictions/                           # prediction / CEMP-candidate outputs
  spatial_filtering_unique/              # cross-match & screening intermediates
  final_validated/                       # final catalogs (produced by scripts/04_validate)
  release/                               # CDS-ready release catalog (produced by scripts/06_release)
  outputs/                               # training checkpoints & logs
```

External inputs that are not produced by this repository (you must obtain them): the
trained checkpoint, LAMOST DR12 spectra/metadata, the APOGEE DR17 catalog, the GALAH
catalog, and the VizieR cross-match result CSV returned after uploading the prepared
upload file.

## 1. Core screening chain

Each step reads the file listed under "input" and writes the file under "output", feeding
the next step. Counts are from the paper's screening chain (Tab. search_chain).

| # | Step (input → output) | Script | Count |
|---|-----------------------|--------|-------|
| 1 | Predict 4 params over DR12; apply the CEMP parameter criterion → `predictions/dr12_cemp_all.csv` | `scripts/02_predict/01_predict_lgadnet_dr12.py` | 331,554 |
| 2 | Deduplicate by LAMOST `DESIG` (keep the best observation) → `predictions/dr12_cemp_unique_by_desig.csv` | `scripts/02_predict/02_dedup_cemp_by_desig.py` | — |
| 3 | High-latitude pre-filter \|b\| > 30° → `spatial_filtering_unique/cemp_unique_b_greater_30.csv` | `scripts/02_predict/03_cemp_spatial_prefilter.py` | 81,471 |
| 4 | Prepare the VizieR upload file → `spatial_filtering_unique/vizier_upload_with_id.csv` | `scripts/02_predict/04_prepare_vizier_upload.py` | — |
| 5 | *(manual)* upload to VizieR; save the returned cross-match result as `vizier_crossmatch_result.csv` | — | — |
| 6 | De-duplicate Gaia double matches (one Gaia source per candidate) → `vizier_dedup_double.csv` | `scripts/03_gaia_crossmatch/01_dedup_vizier_double.py` | 74,233 |
| 7 | Query Bailer-Jones geometric distances → `bailer_jones_distance_for_candidates.csv` | `scripts/03_gaia_crossmatch/02_query_bailer_jones_distance.py` | — |
| 8 | Merge candidates × distances; kinematic filter \|Z_{BJ,lo}\| > 3 kpc and `V_tan,BJ,lo > 180 km/s` → `cemp_bj_robust_candidates_6928.csv` | `scripts/03_gaia_crossmatch/03_bj_phase_space.py` | 6,928 |
| 9 | CMD position check (retain `(BP-RP)_0 ≥ 0.4`, `M_G,BJ,lo ≤ 7.0`) → `cemp_bj_robust_cmd_flags.csv` | `scripts/04_validate/01_plot_bj_robust_gaia_cmd.py` | 4,569 |
| 10 | APOGEE external metallicity constraint (`[Fe/H] ≥ -1` removes metal-rich matches) → `cemp_bj_robust_after_apogee.csv` | `scripts/04_validate/02_apply_apogee_constraint.py` | 4,568 |
| 11 | Compute EGP for the final candidates → `cemp_bj_robust_after_apogee_with_egp.csv` | `scripts/04_validate/03_calculate_egp.py` | — |
| 12 | Export the final catalogs (full / CMD-retained / 16-row sample) → `final_validated/cemp_final_{full,cmdretained,catalog_sample}.csv` | `scripts/04_validate/04_export_final_catalog.py` | — |
| 13 | Build the CDS-ready release catalog (publication column names + uncertainty columns) → `release/cemp_dr12_catalog.csv` | `scripts/06_release/01_build_release_catalog.py` | — |

The screening chain reduces **331,554 → 81,471 → 74,233 → 6,928 → 4,569 → 4,568**,
yielding the final high-confidence CEMP candidate sample.

## 2. Result validation & paper figures

- **Training**: `scripts/01_train/01_prepare_training_dataset.py` builds
  `lgadnet_dataset/{selected_samples,train_labels,val_labels,train_features,val_features}`;
  `scripts/01_train/02_train_lgadnet.py` trains the four-parameter model.
- **Model performance**:
  - `scripts/05_analysis/01_evaluate_regression.py` — validation-set MAE/RMSE/R² and the
    prediction-comparison scatter figure.
  - `scripts/05_analysis/02_evaluate_multiclass.py` — derived three-class
    (other / MP-no-CEMP / CEMP) accuracy metrics.
- **EGP for the training samples** (used by the EGP and CH G-band figures):
  `scripts/05_analysis/03_calculate_egp_for_datasets.py` → `*_with_egp.csv`.
- **Figures**:
  - `scripts/05_analysis/04_galah_lamost_validation.py` — GALAH external-catalog comparison.
  - `scripts/05_analysis/05_plot_cemp_parameter_density.py` — parameter-space density of the
    final CMD-retained sample.
  - `scripts/05_analysis/06_plot_egp_comparison.py` — EGP distribution, training-CEMP vs final.
  - `scripts/05_analysis/07_plot_ch_gband_response.py` — CH G-band absorption profiles.
  - `scripts/05_analysis/08_plot_representative_spectra.py` — representative spectra.

## Final sample

The final 4,568 candidates include **445** with `[Fe/H] < -2.5`, **7** with `[Fe/H] < -3.0`,
**1,320** with `[C/Fe] > 1.0`, and **51** with `[C/Fe] > 2.0`.

## Data availability

- Screening/intermediate and final catalogs, and the dataset files, are provided with the
  GitHub release; scripts run from the `LGADNET_ROOT` layout above.
- The complete final CEMP candidate catalog (4,568 rows) is released via **CDS/VizieR**.
  Each released column is described in the accompanying ReadMe.

## Requirements

Python 3.10+, `numpy`, `pandas`, `astropy`, `scipy`, `scikit-learn`, `torch`,
`matplotlib`, `tqdm`.

## Citation

Please cite the paper when you use this work or the released catalog.