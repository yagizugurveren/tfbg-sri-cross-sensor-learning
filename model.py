from pathlib import Path

import numpy as np
import pandas as pd
import torch

from network import MultiScaleResidualCNN
from preprocessing import load_dataset
from training import get_device, input_scale, predict, to_tensor, train_model


# settings
HERE = Path(__file__).resolve().parent
DATA = {
    "development": HERE.parent / "data-7-8-degrees_true",
    "external_7_8": HERE.parent / "external-validation-dataset_true_angles_7_8_true",
    "external_6_9": HERE.parent / "real-external-dataset",
}
OUTPUT = HERE / "results"
SEEDS = [11, 29, 47, 71, 89]
EPOCHS = 25

NEW_ANGLE_SENSORS = ["external_6deg_S1", "external_6deg_S2", "external_9deg_S1", "external_9deg_S2"]
TRAINING_SRI_RANGE = (1.354, 1.360)


def dev_sensors(angle):
    return [f"{angle:g}deg_S{i}" for i in range(1, 5)]


def build_folds():
    folds = []

    # same angle: hold out one sensor, the next one picks the checkpoint, the other two train
    for angle in (7.0, 8.0):
        sensors = dev_sensors(angle)
        for i, held_out in enumerate(sensors):
            val = sensors[(i + 1) % 4]
            folds.append({
                "name": f"{angle:g}_to_{angle:g}_holdout_{held_out}",
                "direction": f"{angle:g}_to_{angle:g}",
                "train": [s for s in sensors if s not in (held_out, val)],
                "val": val,
                "test": [held_out, f"external_{angle:g}deg_S1"],
            })

    # cross angle: S1-S3 train, S4 picks the checkpoint, everything at the other angle is tested
    for source, target in ((7.0, 8.0), (8.0, 7.0)):
        sensors = dev_sensors(source)
        folds.append({
            "name": f"{source:g}_to_{target:g}",
            "direction": f"{source:g}_to_{target:g}",
            "train": sensors[:3],
            "val": sensors[3],
            "test": dev_sensors(target) + [f"external_{target:g}deg_S1"] + NEW_ANGLE_SENSORS,
        })
    return folds


def rows_of(samples, sensors):
    return samples.index[samples["sensor"].isin(sensors)].to_numpy()


def load_or_train(checkpoint, samples, spectra, train, val, seed, device):
    model = MultiScaleResidualCNN().to(device)

    # skip training if this model was already trained
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model"])
        return model, saved["normalization"]

    # input scale and SRI standardization come from the training sensors only
    sri = samples["sri"].to_numpy(dtype=np.float64)
    scale = input_scale(spectra[train])
    y_mean = float(np.mean(sri[train]))
    y_std = float(max(np.std(sri[train]), 1e-8))
    sensor_codes = pd.factorize(samples.loc[train, "sensor"], sort=True)[0]

    model = train_model(
        to_tensor(spectra[train], scale, device),
        torch.as_tensor((sri[train] - y_mean) / y_std, dtype=torch.float32, device=device),
        sri[train],
        sensor_codes,
        to_tensor(spectra[val], scale, device),
        (sri[val] - y_mean) / y_std,
        scale, seed, EPOCHS, device,
    )
    normalization = {"scale": scale, "y_mean": y_mean, "y_std": y_std}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "normalization": normalization}, checkpoint)
    return model, normalization


def compute_metrics(predictions):
    rows = []
    for (direction, tested_on, sri_range), g in predictions.groupby(["direction", "tested_on", "sri_range"], sort=False):
        truth = g["sri"].to_numpy()
        error = g["predicted_sri"].to_numpy() - truth
        rows.append({
            "direction": direction,
            "tested_on": tested_on,
            "sri_range": sri_range,
            "samples": len(g),
            "models_per_prediction": int(g["models"].iloc[0]),
            "mae_riu": float(np.mean(np.abs(error))),
            "rmse_riu": float(np.sqrt(np.mean(error**2))),
            "max_abs_error_riu": float(np.max(np.abs(error))),
            "bias_riu": float(np.mean(error)),
            "r2": float(1.0 - np.sum(error**2) / np.sum((truth - truth.mean()) ** 2)),
        })
    return pd.DataFrame(rows)


def main():
    device = get_device()
    print(f"Device: {device}", flush=True)
    samples, spectra = load_dataset(DATA, OUTPUT / "processed")

    # train every fold with every seed and predict its test sensors
    records = []
    for fold in build_folds():
        train = rows_of(samples, fold["train"])
        val = rows_of(samples, [fold["val"]])
        test = rows_of(samples, fold["test"])
        for seed in SEEDS:
            print(f"Fold {fold['name']}, seed {seed}", flush=True)
            checkpoint = OUTPUT / "checkpoints" / f"{fold['name']}_seed{seed}.pt"
            model, norm = load_or_train(checkpoint, samples, spectra, train, val, seed, device)
            pred = predict(model, to_tensor(spectra[test], norm["scale"], device))
            records.append(pd.DataFrame({
                "direction": fold["direction"],
                "fold": fold["name"],
                "seed": seed,
                "sample": test,
                "predicted_sri": pred * norm["y_std"] + norm["y_mean"],
            }))
    members = pd.concat(records, ignore_index=True)

    # ensemble = mean over all models of a direction that predicted the same spectrum
    predictions = (
        members.groupby(["direction", "sample"], sort=False)
        .agg(predicted_sri=("predicted_sri", "mean"), models=("predicted_sri", "size"))
        .reset_index()
        .join(samples[["sensor", "sri", "repeat"]].rename(columns={"sensor": "tested_on"}), on="sample")
    )
    inside = predictions["sri"].between(TRAINING_SRI_RANGE[0] - 1e-9, TRAINING_SRI_RANGE[1] + 1e-9)
    predictions["sri_range"] = np.where(inside, "inside", "outside")
    predictions["abs_error_riu"] = (predictions["predicted_sri"] - predictions["sri"]).abs()
    metrics = compute_metrics(predictions)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    members.to_csv(OUTPUT / "predictions_per_model.csv", index=False)
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)

    table = metrics[metrics["sri_range"] == "inside"].pivot(index="tested_on", columns="direction", values="mae_riu") * 1e4
    print("\nEnsemble MAE inside the training SRI range (1e-4 RIU)", flush=True)
    print(table.round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
