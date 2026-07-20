import zipfile
from io import BytesIO
from models.ml_task import MLTask, TaskStatus

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
    "7753de0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _zip_bytes(name: str = "slice_1.tif") -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, TINY_PNG)
    return buf.getvalue()


def test_upload_zip_creates_task(client, session, ml_model, auth_user, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    response = client.post(
        "/api/predict/upload",
        headers=auth_user["headers"],
        data={
            "ml_model_id": str(ml_model.id),
            "patient_age": "60",
            "patient_gender": "2",
        },
        files={"image": ("patient.zip", _zip_bytes(), "application/zip")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert "task_id" in payload
    assert payload["patient_age"] == 60
    assert payload["patient_gender"] == "2"
    assert payload["image_url"].endswith(".zip")


def test_upload_accepts_single_image(client, ml_model, auth_user, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    response = client.post(
        "/api/predict/upload",
        headers=auth_user["headers"],
        data={"ml_model_id": str(ml_model.id)},
        files={"image": ("slice.png", b"not-a-real-png", "image/png")},
    )
    assert response.status_code == 201
    assert response.json()["image_url"].endswith(".png")


def test_upload_rejects_unsupported(client, ml_model, auth_user, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    response = client.post(
        "/api/predict/upload",
        headers=auth_user["headers"],
        data={"ml_model_id": str(ml_model.id)},
        files={"image": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_requires_auth(client, ml_model, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    response = client.post(
        "/api/predict/upload",
        data={"ml_model_id": str(ml_model.id)},
        files={"image": ("patient.zip", _zip_bytes(), "application/zip")},
    )
    assert response.status_code in (401, 403)


def test_get_task_and_history_after_worker_update(
    client, session, ml_model, auth_user, tmp_path, monkeypatch
):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    create = client.post(
        "/api/predict/upload",
        headers=auth_user["headers"],
        data={
            "ml_model_id": str(ml_model.id),
            "patient_age": "44",
            "patient_gender": "1",
        },
        files={"image": ("case.zip", _zip_bytes(), "application/zip")},
    )
    assert create.status_code == 201
    task_id = create.json()["task_id"]

    task = session.get(MLTask, task_id)
    assert task is not None
    assert task.status == TaskStatus.CREATED

    task.status = TaskStatus.COMPLETED
    task.result_mask_path = str(tmp_path / "case_mask.png")
    task.overlay_image_path = str(tmp_path / "case_overlay.png")
    task.similarity_cases = '[{"patient_id":"TCGA_CS_0001","score":0.91}]'
    task.error_message = None
    session.add(task)
    session.commit()

    status_resp = client.get(
        f"/api/predict/{task_id}",
        headers=auth_user["headers"],
    )
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "completed"
    assert body["result_mask_path"].endswith("case_mask.png")
    assert body["overlay_image_path"].endswith("case_overlay.png")
    assert "TCGA_CS_0001" in body["similarity_cases"]

    mine = client.get("/api/predict/tasks/mine", headers=auth_user["headers"])
    assert mine.status_code == 200
    assert any(item["id"] == task_id for item in mine.json())

    review = client.post(
        f"/api/predict/{task_id}/review",
        headers=auth_user["headers"],
        json={"decision": "accepted"},
    )
    assert review.status_code == 200
    assert review.json()["doctor_review"] == "accepted"


def test_get_task_not_found(client, auth_user):
    response = client.get("/api/predict/99999", headers=auth_user["headers"])
    assert response.status_code == 404
