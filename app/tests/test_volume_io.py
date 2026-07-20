from pathlib import Path
import zipfile
import pytest
from mri_preprocessing import preprocess_volume
from volume_io import (
    drop_edge_slices,
    extract_zip,
    list_mri_slice_paths,
    prepare_volume_from_zip,
)


def test_extract_and_list_slices(tmp_path: Path):
    patient = tmp_path / "patient"
    patient.mkdir()
    for i in range(5):
        (patient / f"TCGA_CS_4944_20010208_{i}.tif").write_bytes(b"not-a-real-tif")
    (patient / "TCGA_CS_4944_20010208_2_mask.tif").write_bytes(b"mask")

    zip_path = tmp_path / "case.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in patient.iterdir():
            zf.write(p, arcname=f"TCGA_CS_4944_20010208/{p.name}")

    dest = tmp_path / "extracted"
    extract_zip(zip_path, dest)
    slices = list_mri_slice_paths(dest)
    assert len(slices) == 5
    assert all("_mask" not in p.name for p in slices)
    kept = drop_edge_slices(slices)
    assert len(kept) == 3
    assert kept[0].name.endswith("_1.tif")


def test_reject_path_traversal(tmp_path: Path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "x")

    with pytest.raises(ValueError, match="Небезопасный"):
        extract_zip(zip_path, tmp_path / "out")


def test_prepare_volume_from_real_patient(tmp_path: Path):
    data_root = Path(__file__).resolve().parents[2] / "kaggle_3m"
    patient_dir = data_root / "TCGA_CS_4944_20010208"
    if not patient_dir.exists():
        pytest.skip("kaggle_3m not present")

    zip_path = tmp_path / "TCGA_CS_4944.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in patient_dir.glob("*.tif"):
            if "_mask" in p.name:
                continue
            zf.write(p, arcname=f"{patient_dir.name}/{p.name}")

    volume_raw, kept, all_paths = prepare_volume_from_zip(zip_path, tmp_path / "vol")
    assert len(all_paths) >= 3
    assert volume_raw.shape[0] == len(kept)
    assert volume_raw.dtype.kind == "f"

    volume = preprocess_volume(volume_raw, 160)
    assert volume.shape[0] == len(kept)
    assert volume.shape[1:] == (160, 160, 3)
    assert 0.0 <= float(volume.min()) and float(volume.max()) <= 1.0 + 1e-5
