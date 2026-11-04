import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from network import MultiScaleResidualCNN
from preprocessing import U_GRID


# only this part of the u grid is shown to the model
CORE_WINDOW = (-0.050, -0.0180)

# augmentation
GAIN_RANGE = (0.4, 1.0)
MAX_FLOOR = 0.35
MAX_NOISE = 0.20
MAX_SHIFT = 0.015
MAX_STRETCH = 0.015
CONSISTENCY_WEIGHT = 0.10

# optimisation
BATCH_SIZE = 256
LEVELS_PER_BATCH = 8
MAX_BATCHES_PER_EPOCH = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5


def get_device():
    # GPU if there is one, otherwise CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def input_scale(spectra):
    # median RMS of the training spectra
    rms = np.sqrt(np.mean(np.square(spectra.astype(np.float64)), axis=1))
    return max(float(np.median(rms)), 1e-8)


def to_tensor(spectra, scale, device):
    scaled = np.clip(spectra / scale, -12.0, 12.0).astype(np.float32)
    return torch.from_numpy(scaled)[:, None, :].to(device)


def keep_core_window(x):
    # zero everything outside the core window
    inside = (U_GRID >= CORE_WINDOW[0]) & (U_GRID <= CORE_WINDOW[1])
    return x * torch.as_tensor(inside, dtype=x.dtype, device=x.device)


def degrade_intensity(x, scale):
    # random gain, offset and noise per spectrum (training only)
    shape = (x.shape[0], 1, 1)
    gain = GAIN_RANGE[0] + (GAIN_RANGE[1] - GAIN_RANGE[0]) * torch.rand(shape, device=x.device)
    floor = MAX_FLOOR * torch.rand(shape, device=x.device)
    noise = MAX_NOISE * torch.rand(shape, device=x.device)
    raw = x * scale
    degraded = gain * raw + floor + noise * torch.randn_like(raw)
    return degraded.clamp_min(0.0) / scale


def shift_and_stretch(x):
    # random +-1.5% shift and stretch along the wavelength axis
    batch, _, points = x.shape
    shift = (2.0 * torch.rand(batch, device=x.device) - 1.0) * MAX_SHIFT
    stretch = (2.0 * torch.rand(batch, device=x.device) - 1.0) * MAX_STRETCH
    coord = torch.linspace(-1.0, 1.0, points, device=x.device).view(1, points)
    grid = torch.zeros(batch, 1, points, 2, device=x.device)
    grid[..., 0] = (coord * (1.0 + stretch[:, None]) + shift[:, None])[:, None, :]
    warped = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped.squeeze(2)


def make_batches(sri, sensor, rng):
    # each batch: 8 random SRI levels, same number of spectra from every sensor
    levels = np.unique(sri)
    sensors = np.unique(sensor)
    n_levels = min(LEVELS_PER_BATCH, levels.size)
    per_cell = max(1, BATCH_SIZE // (sensors.size * n_levels))
    n_batches = min(math.ceil(sri.size / BATCH_SIZE), MAX_BATCHES_PER_EPOCH)

    batches = []
    for _ in range(n_batches):
        batch = []
        for level in rng.choice(levels, size=n_levels, replace=False):
            for s in sensors:
                idx = np.flatnonzero((sri == level) & (sensor == s))
                batch.extend(rng.choice(idx, size=per_cell, replace=idx.size < per_cell))
        batch = np.asarray(batch)
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def sensor_balanced_loss(pred, target, sensor):
    # smooth L1, averaged per sensor first so every sensor counts the same
    loss = F.smooth_l1_loss(pred, target, beta=0.25, reduction="none")
    return torch.stack([loss[sensor == s].mean() for s in torch.unique(sensor)]).mean()


def predict(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 1024):
            out.append(model(keep_core_window(x[i:i + 1024])).cpu().numpy())
    return np.concatenate(out)


def train_model(train_x, train_y, train_sri, train_sensor, val_x, val_y, scale, seed, epochs, device):
    set_seed(seed)
    model = MultiScaleResidualCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sensor_codes = torch.as_tensor(train_sensor, device=device)

    best_state, best_error, waiting = None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        for batch in make_batches(train_sri, train_sensor, rng):
            idx = torch.as_tensor(batch, device=device)
            x = keep_core_window(degrade_intensity(train_x[idx], scale))
            y = train_y[idx]
            sensor = sensor_codes[idx]

            # same loss on the original and on a shifted/stretched copy
            pred = model(x)
            pred_warped = model(shift_and_stretch(x))
            regression = 0.5 * (sensor_balanced_loss(pred, y, sensor) + sensor_balanced_loss(pred_warped, y, sensor))
            # the two predictions should agree
            consistency = F.smooth_l1_loss(pred_warped, pred, beta=0.10)
            loss = regression + CONSISTENCY_WEIGHT * consistency

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # early stopping on the validation sensor
        val_error = float(np.mean(np.abs(predict(model, val_x) - val_y)))
        print(f"  epoch {epoch + 1:2d}  val MAE (standardized) {val_error:.4f}", flush=True)
        if val_error < best_error - 1e-7:
            best_state, best_error, waiting = copy.deepcopy(model.state_dict()), val_error, 0
        else:
            waiting += 1
        if waiting >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model
