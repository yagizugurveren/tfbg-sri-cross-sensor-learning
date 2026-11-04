# Cross-sensor SRI prediction with reflection-mode TFBGs

Python implementation of a fixed-branch multiscale residual convolutional neural network for surrounding refractive index (SRI) prediction from reflection-mode tilted fiber Bragg grating (TFBG) spectra. Associated article: *Generalizable full-spectrum learning for recalibration-free refractometry with reflection-mode tilted fiber Bragg gratings*, DOI: 10.1038/s41598-026-79300-9.

## Contents

`preprocessing.py` prepares Bragg-relative, mode-equalized spectra; `network.py` defines the CNN; `training.py` implements training and inference; and `model.py` defines the sensor-disjoint folds and evaluation workflow. The default uses CUDA when available and otherwise CPU.

## Data and reproducibility

The complete workflow expects all eight development sensors at 7° and 8°, both independent 7° and 8° test sensors, and four additional 6° and 9° sensors in the original project layouts. A representative six-sensor subset is being deposited separately; it is not sufficient to retrain the full sensor-disjoint matrix. The exported two-column CSV files in that subset are not in the original multi-channel layout expected by `preprocessing.py` and need an adapter for direct use with this code. The full dataset is available from the corresponding author upon reasonable request.

This package does not include trained weights, predictions, or the full preprocessed dataset. `model.py` searches for the original data directories next to the repository and otherwise cannot run the full workflow; adjust its paths only when the complete original datasets are available. The script runs the specified 7°/8° fold matrix and evaluates the new-angle sensors; the separate unseen-SRI and new-angle retraining studies require additional workflows.

## Installation and execution

Install Python dependencies using `pip install -r requirements.txt`. With the complete datasets in the required layout, run `python model.py`. Outputs are written to `results/`. Training can take substantial time; its precise results may depend on software versions and hardware.

The dataset DOI, software license, and dependency versions should be added when confirmed by the authors.
