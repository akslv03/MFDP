from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
os.environ.setdefault("PYTHONPATH", str(ROOT / "app"))

from clinical_catalog import (
    clinical_payload_for_patient,
    default_data_root,
    load_clinical_rows,
    train_validation_split,
)
from mri_preprocessing import preprocess_volume
from model_runtime import INFERENCE_SIZE, MASK_THRESHOLD, MRISegmentationService
from volume_io import load_volume_from_paths


def _slice_index(path: Path) -> int:
    return int(path.name.split(".")[-2].split("_")[4])


def _load_patient_rgb_volume(patient_dir: Path):
    files = sorted(patient_dir.glob("*.tif"), key=_slice_index)
    image_paths = [fp for fp in files if "mask" not in fp.name]
    if len(image_paths) < 3:
        return None, []
    kept = image_paths[1:-1]
    volume = load_volume_from_paths(kept)
    return volume, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "kaggle_3m"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "mri_tumor_cases"))
    parser.add_argument("--weights", default=str(ROOT / "app" / "weights" / "unet_transformer_finetuned_best.pt"))
    parser.add_argument("--max-patients", type=int, default=0, help="0 = all train patients")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        data_root = default_data_root()

    all_patient_ids = sorted(
        p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("TCGA_")
    )
    train_ids, valid_ids = train_validation_split(all_patient_ids)
    print(f"Patients total={len(all_patient_ids)} train={len(train_ids)} valid={len(valid_ids)}")
    print(f"Validation held out (not indexed): {valid_ids}")

    if args.max_patients > 0:
        train_ids = train_ids[: args.max_patients]

    runtime = MRISegmentationService(model_path=args.weights)
    model = runtime._load_model()
    clinical_rows = load_clinical_rows(data_root)

    points = []
    vector_size = None

    for idx, patient_id in enumerate(tqdm(train_ids, desc="Indexing train")):
        patient_dir = data_root / patient_id
        volume_raw, kept_paths = _load_patient_rgb_volume(patient_dir)
        if volume_raw is None:
            print(f"skip {patient_id}: empty volume")
            continue

        volume = preprocess_volume(volume_raw, INFERENCE_SIZE)
        tensor = runtime._volume_tensor(volume)
        patient_emb, _probs, binaries = runtime._weighted_volume_embedding(model, tensor)
        patient_emb = np.asarray(patient_emb, dtype=np.float32)
        vector_size = int(patient_emb.shape[0])

        areas = binaries.reshape(binaries.shape[0], -1).sum(axis=1)
        best_i = int(np.argmax(areas)) if areas.size else 0
        best_slice_filename = kept_paths[best_i].name if kept_paths else None

        payload = clinical_payload_for_patient(patient_id, clinical_rows)
        payload.update(
            {
                "split": "train",
                "total_slices": int(volume.shape[0]),
                "estimated_tumor_volume": float(areas.sum()),
                "best_threshold_used": float(MASK_THRESHOLD),
                "best_slice_filename": best_slice_filename,
            }
        )
        points.append(
            PointStruct(id=idx, vector=patient_emb.tolist(), payload=payload)
        )

    if not points or vector_size is None:
        raise SystemExit("No points indexed")

    client = QdrantClient(url=args.qdrant_url, timeout=60.0)
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.upsert(collection_name=args.collection, wait=True, points=points)
    print(f"Upserted {len(points)} train patients into {args.collection} @ {args.qdrant_url}")
    print(f"Vector size={vector_size}. Validation patients excluded: {len(valid_ids)}")


if __name__ == "__main__":
    main()
