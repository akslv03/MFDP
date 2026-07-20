"""Просмотр исторического МРТ-кейса из kaggle_3m."""

from __future__ import annotations
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from clinical_catalog import clinical_payload_for_patient, default_data_root
from volume_io import IMAGE_SUFFIXES, _SLICE_INDEX_RE, _is_mask_name, _slice_sort_key

_PATIENT_ID_RE = re.compile(r"^TCGA_[A-Za-z0-9_]+$")


def is_safe_patient_id(patient_id: str) -> bool:
    return bool(patient_id and _PATIENT_ID_RE.match(patient_id))


def resolve_patient_dir(patient_id: str, data_root: Path | None = None) -> Path:
    if not is_safe_patient_id(patient_id):
        raise ValueError("Некорректный patient_id")
    root = (data_root or default_data_root()).resolve()
    patient_dir = (root / patient_id).resolve()
    if not str(patient_dir).startswith(str(root)):
        raise ValueError("Некорректный patient_id")
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Пациент не найден: {patient_id}")
    return patient_dir


def _mask_path_for(slice_path: Path) -> Optional[Path]:
    candidate = slice_path.with_name(f"{slice_path.stem}_mask{slice_path.suffix}")
    return candidate if candidate.is_file() else None


def list_patient_slices(patient_id: str, data_root: Path | None = None) -> List[Dict[str, Any]]:
    patient_dir = resolve_patient_dir(patient_id, data_root=data_root)
    paths = sorted(
        (
            p
            for p in patient_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and not _is_mask_name(p.name)
        ),
        key=_slice_sort_key,
    )
    slices: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        match = _SLICE_INDEX_RE.search(path.name)
        slices.append(
            {
                "index": idx,
                "filename": path.name,
                "slice_number": int(match.group(1)) if match else idx,
                "has_mask": _mask_path_for(path) is not None,
            }
        )
    return slices


def get_case_detail(patient_id: str, data_root: Path | None = None) -> Dict[str, Any]:
    resolve_patient_dir(patient_id, data_root=data_root)
    slices = list_patient_slices(patient_id, data_root=data_root)
    clinical = clinical_payload_for_patient(patient_id)
    return {
        "patient_id": patient_id,
        "total_slices": len(slices),
        "slices": slices,
        "clinical": clinical,
    }


def resolve_slice_path(
    patient_id: str,
    filename: str,
    data_root: Path | None = None,
) -> Path:
    patient_dir = resolve_patient_dir(patient_id, data_root=data_root)
    name = Path(filename).name
    if name != filename or ".." in name or "/" in name or "\\" in name:
        raise ValueError("Некорректное имя файла")
    path = (patient_dir / name).resolve()
    if not str(path).startswith(str(patient_dir.resolve())):
        raise ValueError("Некорректное имя файла")
    if not path.is_file():
        raise FileNotFoundError(f"Срез не найден: {filename}")
    return path


def slice_to_png_bytes(
    patient_id: str,
    filename: str,
    *,
    with_mask: bool = False,
    data_root: Path | None = None,
) -> bytes:
    path = resolve_slice_path(patient_id, filename, data_root=data_root)
    img = Image.open(path)

    if img.mode == "L":
        gray = img
    else:
        rgb = img.convert("RGB")
        r, g, b = rgb.split()
        gray = g

    if with_mask:
        mask_path = _mask_path_for(path)
        if mask_path is not None:
            mask = Image.open(mask_path)
            if mask.mode != "L":
                mask = mask.convert("L")
            base = Image.merge("RGB", (gray, gray, gray))
            mask_bin = mask.point(lambda x: 255 if x > 0 else 0)
            overlay = Image.new("RGB", base.size, (255, 0, 0))
            base.paste(overlay, mask=mask_bin)
            out = base
        else:
            out = gray
    else:
        out = gray

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
