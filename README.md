# Yet-Another-Exoplanet-Classifier

A convolutional neural network ablation study investigating whether centroid motion time series from TESS photometry improves automated classification of exoplanet transit candidates versus eclipsing binary false positives.

---

## Overview

This repository implements a three-model ablation study using phase-folded TESS light curves to classify TOI (TESS Object of Interest) candidates as either planet candidates or false positives (eclipsing binaries). The central question is whether adding centroid motion channels — MOM_CENTR1 and MOM_CENTR2 from the TESS SPOC pipeline — alongside flux improves classification performance over flux alone.

All three models share an identical dual-branch 1D CNN architecture and training configuration, differing only in their input channels. 

---

## Pipeline

### Step 1 — Build the target manifest
```bash
python getExamples.py
```
Queries the TESS TOI catalogue via ExoFOP and produces `classified_targets.csv` with stratified positive (planet candidate) and negative (false positive / eclipsing binary) labels.

### Step 2 — Download and process light curves
```bash
python getInputData.py
```
Runs a three-phase async pipeline:
- **Phase 1:** Queries MAST via `lightkurve` to find available SPOC FITS files for each target
- **Phase 2:** Downloads FITS files concurrently via `aiohttp` (15-connection semaphore)
- **Phase 3:** Processes each target — quality masking, sigma-clipping, phase folding, OOT normalisation, median binning to 1,000 phase bins — and writes `tess_training_data.csv`

Each row in the output CSV contains 3,006 columns: 6 metadata fields followed by 1,000 flux bins (`f_0`…`f_999`), 1,000 RA centroid bins (`m1_0`…`m1_999`), and 1,000 Dec centroid bins (`m2_0`…`m2_999`).

### Step 3 — Train the models
Run each model script independently:
```bash
python theModel_fluxOnly.py        # Model A — flux data only 
python theModel.py                 # Model B — flux + centroid data 
python theModel_centroidOnly.py    # Model C — centroid data only 
```

Each script performs 5-fold stratified cross-validation with per-fold threshold optimisation, followed by 5-fold ensemble prediction. Results are saved to their respective `results_*/` directory as a confusion matrix PNG and a metrics report text file.

### Step 4 — Run the Wilcoxon test
```bash
python wilcoxon_test.py
```
Reads per-fold AUC values from each model's metrics report and runs paired Wilcoxon signed-rank tests comparing Model A vs B and Model A vs C. Prints and saves results to `results/wilcoxon_results.txt`.

---

## Model Architecture

A dual-branch shallow 1D CNN implemented in TensorFlow/Keras:

- **Global branch** — receives the full input sequence (1,000 bins, all channels). Three Conv1D blocks (32→64→128 filters, kernel size 5) with BatchNorm, ReLU, and MaxPooling, followed by GlobalAveragePooling → 128-dim vector.
- **Local branch** — receives only the central 200 bins of the flux channel (the transit window). Two Conv1D blocks (32→64 filters) followed by GlobalAveragePooling → 64-dim vector.
- **Classification head** — Concatenate(128+64) → Dense(128) → Dropout(0.5) → Dense(32) → Dropout(0.5) → Dense(1, sigmoid)

Training uses focal loss (γ=2.0, label smoothing=0.1), Adam optimiser, linear LR warmup, ReduceLROnPlateau, EarlyStopping, class weights {FP: 2.0, Planet: 1.0}, and three physically motivated augmentations (phase jitter, Gaussian noise, flux scaling).

---

## Results

| Model | Input | CV AUC-ROC |
|---|---|---|
| A — Flux Only | (N, 1000, 1) | 0.7268 ± 0.0216 |
| B — Flux + Centroid | (N, 1000, 3) | 0.7287 ± 0.0216 |
| C — Centroid Only | (N, 1000, 2) | 0.6151 ± 0.0274 |

Wilcoxon test (A vs B): W=6.0, p=0.8125 | Adding centroid channels produces no meaningful improvement over flux alone.

Wilcoxon test (A vs C): W=0.0, p=0.0625 | Which is narrowly below the power ceiling of the test at n=5 (minimum achievable p=0.0625). Model A outperformed Model C in all five folds.

---

## Requirements

Install with:
```bash
pip install tensorflow scikit-learn numpy pandas scipy lightkurve aiohttp astropy tqdm matplotlib
```

---

## Dataset

- **Source:** TESS TOI catalogue via ExoFOP, light curves from MAST SPOC pipeline
- **Size:** 2,023 examples — 962 false positives (label 0), 1,061 planet candidates (label 1)
- **Input shape:** (N, 1000, 3) — phase-folded, OOT-normalised, median-binned
