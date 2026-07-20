from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from case_browser import get_case_detail, is_safe_patient_id, slice_to_png_bytes


def _write_tif(path: Path, value: int = 40) -> None:
    arr = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def fake_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "kaggle_3m"
    patient = root / "TCGA_CS_4941_19960909"
    patient.mkdir(parents=True)
    _write_tif(patient / "TCGA_CS_4941_19960909_1.tif", 20)
    _write_tif(patient / "TCGA_CS_4941_19960909_1_mask.tif", 0)
    _write_tif(patient / "TCGA_CS_4941_19960909_2.tif", 80)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(mask, mode="L").save(patient / "TCGA_CS_4941_19960909_2_mask.tif")

    csv_path = root / "data.csv"
    csv_path.write_text(
        "Patient,RNASeqCluster,MethylationCluster,miRNACluster,CNCluster,"
        "RPPACluster,OncosignCluster,COCCluster,histological_type,"
        "neoplasm_histologic_grade,tumor_tissue_site,laterality,tumor_location,"
        "gender,age_at_initial_pathologic,race,ethnicity,death01\n"
        "TCGA_CS_4941,1,2,3,4,5,6,7,astrocytoma,G2,Brain,Left,Frontal,1,55,white,not hispanic,0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_3M_PATH", str(root))
    return root


def test_safe_patient_id():
    assert is_safe_patient_id("TCGA_CS_4941_19960909")
    assert not is_safe_patient_id("../etc/passwd")
    assert not is_safe_patient_id("TCGA/CS")


def test_get_case_detail(fake_kaggle: Path):
    detail = get_case_detail("TCGA_CS_4941_19960909")
    assert detail["patient_id"] == "TCGA_CS_4941_19960909"
    assert detail["total_slices"] == 2
    assert detail["slices"][0]["has_mask"] is True
    assert detail["clinical"]["age"] == 55
    assert detail["clinical"]["gender"] == "1"


def test_slice_to_png_with_mask(fake_kaggle: Path):
    plain = slice_to_png_bytes(
        "TCGA_CS_4941_19960909",
        "TCGA_CS_4941_19960909_2.tif",
        with_mask=False,
    )
    masked = slice_to_png_bytes(
        "TCGA_CS_4941_19960909",
        "TCGA_CS_4941_19960909_2.tif",
        with_mask=True,
    )
    assert plain[:8] == b"\x89PNG\r\n\x1a\n"
    assert masked[:8] == b"\x89PNG\r\n\x1a\n"
    assert plain != masked
