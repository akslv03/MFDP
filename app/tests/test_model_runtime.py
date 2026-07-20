from unittest.mock import MagicMock, patch
import os
import numpy as np
import pytest
from clinical_catalog import train_validation_split
from mask_utils import postprocess_mask
from similarity_search import search_similar_cases


def test_postprocess_removes_tiny_components():
    mask_prob = np.zeros((40, 40), dtype=np.float32)
    mask_prob[5:20, 5:20] = 0.9
    mask_prob[30, 30] = 0.95

    result = postprocess_mask(mask_prob, threshold=0.20)

    assert result[10, 10] == 1
    assert result[30, 30] == 0


def test_train_validation_split_no_overlap():
    patients = [f"TCGA_CS_{i:04d}_20000101" for i in range(110)]
    train, valid = train_validation_split(patients, seed=42, k_valid=10)
    assert len(valid) == 10
    assert len(train) == 100
    assert set(train).isdisjoint(valid)


def test_missing_weights_raise_clear_error(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from model_runtime import MRISegmentationService

    service = MRISegmentationService(model_path=str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        service._resolve_model_path()
    assert "В папке нет весов модели" in str(exc.value)


def test_predict_rejects_unsupported_format(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from model_runtime import MRISegmentationService

    unsupported = tmp_path / "scan.dcm"
    unsupported.write_bytes(b"x")
    service = MRISegmentationService(model_path=str(tmp_path))
    with pytest.raises(ValueError) as exc:
        service.predict(str(unsupported))
    assert "изображение" in str(exc.value)


def test_weights_file_is_present_for_mvp():
    weights_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "weights",
        "unet_transformer_finetuned_best.pt",
    )
    assert os.path.isfile(weights_path)


def test_similarity_search_returns_json_list():
    hit = MagicMock()
    hit.score = 0.87
    hit.payload = {
        "patient_id": "TCGA_CS_0001",
        "age": 55,
        "gender": "1",
        "split": "train",
    }

    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(points=[hit])

    with patch("similarity_search._client", return_value=client):
        raw = search_similar_cases([0.1, 0.2, 0.3], age=55, gender="1", limit=3)

    assert "TCGA_CS_0001" in raw
    assert "0.87" in raw or "0.870" in raw
