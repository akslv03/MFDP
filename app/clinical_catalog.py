from __future__ import annotations
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

SPLIT_SEED = 42
VALIDATION_PATIENTS_K = 10

CLINICAL_CSV_FIELDS = (
    "Patient",
    "RNASeqCluster",
    "MethylationCluster",
    "miRNACluster",
    "CNCluster",
    "RPPACluster",
    "OncosignCluster",
    "COCCluster",
    "histological_type",
    "neoplasm_histologic_grade",
    "tumor_tissue_site",
    "laterality",
    "tumor_location",
    "gender",
    "age_at_initial_pathologic",
    "race",
    "ethnicity",
    "death01",
)


def default_data_root() -> Path:
    env = os.getenv("KAGGLE_3M_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "kaggle_3m"


def list_patient_dirs(data_root: Path | None = None) -> List[str]:
    root = data_root or default_data_root()
    patients = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("TCGA_")
    )
    return patients


def train_validation_split(
    patients: List[str] | None = None,
    seed: int = SPLIT_SEED,
    k_valid: int = VALIDATION_PATIENTS_K,
) -> tuple[List[str], List[str]]:
    """10 пациентов на валидацию, остальные — train."""
    all_patients = sorted(patients if patients is not None else list_patient_dirs())
    rng = random.Random(seed)
    validation = sorted(rng.sample(all_patients, k=min(k_valid, len(all_patients))))
    train = sorted(set(all_patients).difference(validation))
    return train, validation


def load_clinical_rows(data_root: Path | None = None) -> Dict[str, Dict[str, str]]:
    """Читает data.csv и возвращает словарь по patient id."""
    root = data_root or default_data_root()
    csv_path = root / "data.csv"
    by_patient: Dict[str, Dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("Patient") or "").strip()
            if pid:
                by_patient[pid] = {k: (row.get(k) or "").strip() for k in CLINICAL_CSV_FIELDS}
    return by_patient


def _cell(row: Optional[Dict[str, str]], key: str, default: str = "unknown") -> str:
    if not row:
        return default
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def clinical_payload_for_patient(
    patient_id: str,
    clinical_rows: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    Метаданные пациента для Qdrant.
    """
    rows = clinical_rows if clinical_rows is not None else load_clinical_rows()
    short_id = "_".join(patient_id.split("_")[:3])
    row = rows.get(patient_id) or rows.get(short_id)

    age_raw = _cell(row, "age_at_initial_pathologic", "")
    try:
        age = int(float(age_raw)) if age_raw not in ("", "unknown") else -1
    except ValueError:
        age = -1

    death_raw = _cell(row, "death01", "")
    try:
        death01 = int(float(death_raw)) if death_raw not in ("", "unknown") else -1
    except ValueError:
        death01 = -1

    return {
        "patient_id": patient_id,
        "short_id": short_id,
        "age": age,
        "gender": _cell(row, "gender", "unknown"),
        "histological_type": _cell(row, "histological_type", "unknown"),
        "grade": _cell(row, "neoplasm_histologic_grade", "unknown"),
        "location": _cell(row, "tumor_location", "unknown"),
        "laterality": _cell(row, "laterality", "unknown"),
        "tumor_tissue_site": _cell(row, "tumor_tissue_site", "unknown"),
        "death01": death01,
        "rna_cluster": _cell(row, "RNASeqCluster", "unknown"),
        "meth_cluster": _cell(row, "MethylationCluster", "unknown"),
        "mirna_cluster": _cell(row, "miRNACluster", "unknown"),
        "cn_cluster": _cell(row, "CNCluster", "unknown"),
        "rppa_cluster": _cell(row, "RPPACluster", "unknown"),
        "oncosign_cluster": _cell(row, "OncosignCluster", "unknown"),
        "coc_cluster": _cell(row, "COCCluster", "unknown"),
        "race": _cell(row, "race", "unknown"),
        "ethnicity": _cell(row, "ethnicity", "unknown"),
    }
