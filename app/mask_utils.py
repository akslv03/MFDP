"""Постобработка маски сегментации."""

from __future__ import annotations
import cv2
import numpy as np

MASK_THRESHOLD = 0.20


def postprocess_mask(
    mask_prob: np.ndarray,
    threshold: float = MASK_THRESHOLD,
) -> np.ndarray:
    binary_mask = (mask_prob > threshold).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels > 1:
        cleaned = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 10:
                cleaned[labels == i] = 1
        binary_mask = cleaned

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
