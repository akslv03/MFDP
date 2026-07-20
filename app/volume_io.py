from __future__ import annotations
import re
import zipfile
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MAX_SLICES = 1024
_SLICE_INDEX_RE = re.compile(r"_(\d+)(?:_mask)?\.(?:tif|tiff|png|jpg|jpeg|bmp)$", re.I)


def is_zip_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".zip"


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def _is_mask_name(name: str) -> bool:
    stem = Path(name).stem.lower()
    return stem.endswith("_mask") or "_mask." in name.lower()


def _slice_sort_key(path: Path) -> tuple:
    match = _SLICE_INDEX_RE.search(path.name)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, 0, path.name.lower())


def extract_zip(zip_path: str | Path, dest_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Небезопасный путь внутри ZIP: {info.filename}")
            target = (dest_dir / member).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise ValueError(f"Путь в ZIP выходит за пределы папки: {info.filename}")
        zf.extractall(dest_dir)

    return dest_dir


def list_mri_slice_paths(root: str | Path) -> List[Path]:
    root = Path(root)
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not _is_mask_name(p.name)
    ]
    paths = sorted(paths, key=_slice_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"В папке нет МРТ-срезов ({', '.join(sorted(IMAGE_SUFFIXES))})"
        )
    if len(paths) > MAX_SLICES:
        raise ValueError(f"Слишком много срезов в архиве (> {MAX_SLICES})")
    return paths


def drop_edge_slices(paths: List[Path]) -> List[Path]:
    if len(paths) >= 3:
        return paths[1:-1]
    return paths


def read_slice_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def load_volume_from_paths(paths: List[Path]) -> np.ndarray:
    arrays = [read_slice_rgb(p) for p in paths]
    return np.stack(arrays, axis=0)


def prepare_volume_from_zip(zip_path: str | Path, extract_dir: str | Path):
    extract_dir = Path(extract_dir)
    extract_zip(zip_path, extract_dir)
    all_paths = list_mri_slice_paths(extract_dir)
    kept = drop_edge_slices(all_paths)
    volume = load_volume_from_paths(kept)
    return volume, kept, all_paths


def prepare_volume_from_image(image_path: str | Path):
    path = Path(image_path)
    volume = read_slice_rgb(path)[np.newaxis, ...]
    return volume, [path]
