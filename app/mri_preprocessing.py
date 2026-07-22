from __future__ import annotations
import numpy as np
from PIL import Image

FOREGROUND_FRACTION = 0.1


def _resize_slice(slice_hwc: np.ndarray, size: int) -> np.ndarray:
    channels = []
    for c in range(slice_hwc.shape[-1]):
        img = Image.fromarray(slice_hwc[..., c].astype(np.float32), mode="F")
        img = img.resize((size, size), resample=Image.BICUBIC)
        channels.append(np.asarray(img, dtype=np.float32))
    return np.stack(channels, axis=-1)


def crop_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32).copy()
    peak = float(volume.max()) if volume.size else 0.0
    if peak <= 0.0:
        return volume

    volume[volume < peak * FOREGROUND_FRACTION] = 0.0
    z_projection = volume.max(axis=(1, 2, 3))
    y_projection = volume.max(axis=(0, 2, 3))
    x_projection = volume.max(axis=(0, 1, 3))

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
    resized = [_resize_slice(volume[i], size) for i in range(volume.shape[0])]
    return np.stack(resized, axis=0).astype(np.float32)


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
