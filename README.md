# Cross-sensor SRI prediction with reflection-mode TFBGs

Python implementation of a fixed-branch multiscale residual convolutional neural network for surrounding refractive index (SRI) prediction from reflection-mode tilted fiber Bragg grating (TFBG) spectra. Associated article: *Generalizable full-spectrum learning for recalibration-free refractometry with reflection-mode tilted fiber Bragg gratings*, DOI: 10.1038/s41598-026-79300-9.

## Contents

`preprocessing.py` prepares Bragg-relative, mode-equalized spectra; `network.py` defines the CNN; `training.py` implements training and inference; and `model.py` defines the sensor-disjoint folds and evaluation workflow. The default uses CUDA when available and otherwise CPU.

## Installation and execution

Install Python dependencies using `pip install -r requirements.txt`. With the complete datasets in the required layout, run `python model.py`. Outputs are written to `results/`.


The code is released under the MIT License. The dataset DOI and dependency versions should be added when confirmed by the authors.
