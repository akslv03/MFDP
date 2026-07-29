from __future__ import annotations
import numpy as np
from skimage.transform import resize

FOREGROUND_FRACTION = 0.1


def crop_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    peak = float(volume.max()) if volume.size else 0.0
    if peak <= 0.0:
        return volume

    probe = volume.copy()
    probe[probe < peak * FOREGROUND_FRACTION] = 0.0
    z_projection = probe.max(axis=(1, 2, 3))
    y_projection = probe.max(axis=(0, 2, 3))
    x_projection = probe.max(axis=(0, 1, 3))

    z_nz = np.nonzero(z_projection)[0]
    y_nz = np.nonzero(y_projection)[0]
    x_nz = np.nonzero(x_projection)[0]
    if z_nz.size == 0 or y_nz.size == 0 or x_nz.size == 0:
        return volume

    return volume[
        z_nz.min() : z_nz.max() + 1,
        y_nz.min() : y_nz.max() + 1,
        x_nz.min() : x_nz.max() + 1,
    ]


def pad_volume(volume: np.ndarray) -> np.ndarray:
    height, width = volume.shape[1], volume.shape[2]
    if height == width:
        return volume

    diff = (max(height, width) - min(height, width)) / 2.0
    low, high = int(np.floor(diff)), int(np.ceil(diff))
    if height > width:
        padding = ((0, 0), (0, 0), (low, high), (0, 0))
    else:
        padding = ((0, 0), (low, high), (0, 0), (0, 0))
    return np.pad(volume, padding, mode="constant", constant_values=0)


def resize_volume(volume: np.ndarray, size: int) -> np.ndarray:
    out_shape = (volume.shape[0], size, size, volume.shape[3])
    return resize(
        volume.astype(np.float32),
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    ).astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    p10, p99 = np.percentile(volume, 10), np.percentile(volume, 99)
    volume = np.clip(volume, p10, p99)
    return (volume - volume.min()) / (volume.max() - volume.min() + 1e-5)


def preprocess_volume(volume: np.ndarray, size: int) -> np.ndarray:
    volume = crop_volume(volume)
    volume = pad_volume(volume)
    volume = resize_volume(volume, size)
    return normalize_volume(volume)
