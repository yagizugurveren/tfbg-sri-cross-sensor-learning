import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, maximum_filter1d


# wavelength grid (nm) and Bragg-relative grid u = (wl - bragg) / bragg used as model input
RAW_GRID = np.linspace(1460.0, 1619.992, 2048)
U_GRID = np.linspace(-0.049998782040250656, -0.01803138782443669, 755)

# the Bragg peak is the strongest positive peak in this range for every grating (nm)
BRAGG_WINDOW = (1570.0, 1615.0)

# (sensor name, tilt angle, channel in the file)
FILE_GROUPS = [
    {"dataset": "development", "angle": 7.0,
     "sensors": [("7deg_S1", 7.0, 1), ("7deg_S2", 7.0, 2), ("7deg_S3", 7.0, 3), ("7deg_S4", 7.0, 4)]},
    {"dataset": "development", "angle": 8.0,
     "sensors": [("8deg_S1", 8.0, 1), ("8deg_S2", 8.0, 2), ("8deg_S3", 8.0, 3), ("8deg_S4", 8.0, 4)]},
    {"dataset": "external_7_8", "angle": 7.0,
     "sensors": [("external_7deg_S1", 7.0, 1)]},
    {"dataset": "external_7_8", "angle": 8.0,
     "sensors": [("external_8deg_S1", 8.0, 1)]},
    {"dataset": "external_6_9", "angle": None,
     "sensors": [("external_6deg_S1", 6.0, 1), ("external_6deg_S2", 6.0, 2),
                 ("external_9deg_S1", 9.0, 3), ("external_9deg_S2", 9.0, 4)]},
]
# Only 7deg_S2, 8deg_S1, external_7deg_S1, external_8deg_S1, external_6deg_S1 and external_9deg_S1 are shared.


def files_from_manifest(root, angle):
    manifest = pd.read_csv(root / "manifest.csv")
    manifest = manifest[np.isclose(manifest["angle_deg"], angle)]
    return pd.DataFrame({
        "sri": manifest["sri"].round(4).to_numpy(),
        "repeat": manifest["repeat_id"].astype(int).to_numpy(),
        "path": [str(root / p) for p in manifest["relative_path"]],
    })


def files_from_6_9_folders(root):
    # folders look like .../spectra/label_1-3600/..._rep-001.csv
    rows = []
    for path in sorted(root.glob("*/spectra/label_1-*/*.csv")):
        level = re.search(r"label_1-(\d+)", path.parent.name).group(1)
        repeat = re.search(r"rep-(\d+)", path.name).group(1)
        rows.append({"sri": round(float(f"1.{level}"), 4), "repeat": int(repeat), "path": str(path)})
    return pd.DataFrame(rows)


def list_files(group, roots):
    root = roots[group["dataset"]]
    if group["dataset"] == "external_6_9":
        files = files_from_6_9_folders(root)
    else:
        files = files_from_manifest(root, group["angle"])
    if files.empty:
        raise FileNotFoundError(f"No spectra found in {root}")
    return files.sort_values(["sri", "repeat"]).reset_index(drop=True)


def read_spectrum(path, dataset, channels):
    if dataset == "external_6_9":
        wl_col = "Wavelength"
        names = {ch: f"Insertion_Loss_{ch}" for ch in channels}
        table = pd.read_csv(path, sep="\t", usecols=[wl_col, *names.values()])
    else:
        wl_col = "Wavelenghts"
        names = {ch: f"IntensityCh{ch}" for ch in channels}
        table = pd.read_csv(path, usecols=[wl_col, *names.values()])
    wavelength = table[wl_col].to_numpy(dtype=np.float64)
    intensities = {ch: table[name].to_numpy(dtype=np.float64) for ch, name in names.items()}

    # sanity check
    if np.any(np.diff(wavelength) <= 0) or not all(np.isfinite(v).all() for v in intensities.values()):
        raise ValueError(f"Bad spectrum: {path}")
    return wavelength, intensities


def db_to_linear(db, reference_db):
    return np.power(10.0, np.clip((db - reference_db) / 10.0, -12.0, 12.0))


def find_bragg(wavelength, intensity):
    window = (wavelength >= BRAGG_WINDOW[0]) & (wavelength <= BRAGG_WINDOW[1])
    wl = wavelength[window]
    step = float(np.median(np.diff(wl)))

    # strongest positive peak above a smoothed baseline
    local = intensity[window]
    prominence = local - gaussian_filter1d(local, sigma=max(1.5, 1.0 / step), mode="nearest")
    peak = int(np.argmax(prominence))
    i = int(np.clip(peak, 1, prominence.size - 2))

    # parabola through the 3 points around the peak for sub-sample accuracy
    left, mid, right = prominence[i - 1:i + 2]
    curvature = left - 2.0 * mid + right
    offset = 0.0
    if peak == i and abs(curvature) > 1e-12:
        offset = float(np.clip(0.5 * (left - right) / curvature, -1.0, 1.0))
    return float(wl[i] + offset * step)


def bragg_reference(wavelength, intensity, bragg):
    # flanks on both sides of the Bragg peak (0.6 - 1.0 nm away)
    left = (wavelength >= bragg - 1.0) & (wavelength <= bragg - 0.6)
    right = (wavelength >= bragg + 0.6) & (wavelength <= bragg + 1.0)
    reference_db = float(np.median(np.concatenate([intensity[left], intensity[right]])))

    # noise = robust spread of the flanks after removing a straight baseline
    power = db_to_linear(intensity, reference_db)
    left_power = float(np.median(power[left]))
    right_power = float(np.median(power[right]))
    left_wl = float(np.median(wavelength[left]))
    right_wl = float(np.median(wavelength[right]))
    slope = (right_power - left_power) / (right_wl - left_wl)
    residual = power - (left_power + slope * (wavelength - left_wl))
    flanks = np.concatenate([residual[left], residual[right]])
    noise = 1.4826 * float(np.median(np.abs(flanks - np.median(flanks))))
    return reference_db, max(noise, 1e-10)


def mode_equalize(intensity_db, reference_db, noise):
    power = db_to_linear(intensity_db.astype(np.float64), reference_db)

    # upper continuum -> dip depth
    continuum = gaussian_filter1d(maximum_filter1d(power, size=31, mode="nearest"), sigma=8.0, mode="nearest")
    depth = np.maximum(continuum - power, 0.0)

    # divide each dip by its local envelope so all modes have similar height
    envelope = gaussian_filter1d(maximum_filter1d(depth, size=41, mode="nearest"), sigma=12.0, mode="nearest")
    floor = max(0.05 * float(envelope.max()), 3.0 * noise)
    return np.clip(depth / np.maximum(envelope, floor), 0.0, 3.0).astype(np.float32)


def preprocess(wavelength, intensity):
    resampled = np.interp(RAW_GRID, wavelength, intensity).astype(np.float32)
    bragg = find_bragg(wavelength, intensity)
    reference_db, noise = bragg_reference(wavelength, intensity, bragg)

    relative = np.interp(bragg * (1.0 + U_GRID), RAW_GRID, resampled).astype(np.float32)
    return mode_equalize(relative, reference_db, noise)


def build_dataset(roots):
    tables, spectra = [], []
    for group in FILE_GROUPS:
        files = list_files(group, roots)
        channels = [ch for _, _, ch in group["sensors"]]
        processed = {ch: [] for ch in channels}

        # one file holds several sensors, so read it once
        for n, row in files.iterrows():
            wavelength, intensities = read_spectrum(row["path"], group["dataset"], channels)
            for ch in channels:
                processed[ch].append(preprocess(wavelength, intensities[ch]))
            if n == 0 or (n + 1) % 500 == 0 or n + 1 == len(files):
                print(f"Preprocessing {group['dataset']} {group['angle'] or ''}: {n + 1}/{len(files)}", flush=True)

        for name, angle, ch in group["sensors"]:
            table = files.copy()
            table.insert(0, "sensor", name)
            table.insert(1, "dataset", group["dataset"])
            table.insert(2, "angle", angle)
            tables.append(table)
            spectra.append(np.stack(processed[ch]))
    return pd.concat(tables, ignore_index=True), np.concatenate(spectra)


def load_dataset(roots, folder):
    # reuse the preprocessed data if it is already there
    samples_file = folder / "samples.csv"
    spectra_file = folder / "spectra.npy"
    if samples_file.is_file() and spectra_file.is_file():
        return pd.read_csv(samples_file), np.load(spectra_file)

    samples, spectra = build_dataset(roots)
    folder.mkdir(parents=True, exist_ok=True)
    samples.to_csv(samples_file, index=False)
    np.save(spectra_file, spectra)
    return samples, spectra
