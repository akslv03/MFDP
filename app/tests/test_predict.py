def test_create_ml_task_zip(client, ml_model, auth_user, tmp_path):
    zip_path = tmp_path / "patient.zip"
    zip_path.write_bytes(b"PK fake")

    response = client.post(
        "/api/predict/",
        headers=auth_user["headers"],
        json={
            "ml_model_id": ml_model.id,
            "image_url": str(zip_path),
            "patient_age": 55,
            "patient_gender": "1",
        },
    )

    assert response.status_code == 201
    assert "task_id" in response.json()


def test_create_ml_task_rejects_unsupported(client, ml_model, auth_user):
    response = client.post(
        "/api/predict/",
        headers=auth_user["headers"],
        json={
            "ml_model_id": ml_model.id,
            "image_url": "notes.txt",
            "patient_age": 55,
        },
    )
    assert response.status_code == 422


def test_create_ml_task_accepts_single_image(client, ml_model, auth_user):
    response = client.post(
        "/api/predict/",
        headers=auth_user["headers"],
        json={
            "ml_model_id": ml_model.id,
            "image_url": "slice.png",
            "patient_age": 55,
        },
    )
    assert response.status_code == 201


def test_create_ml_task_requires_auth(client, ml_model):
    response = client.post(
        "/api/predict/",
        json={
            "ml_model_id": ml_model.id,
            "image_url": "patient.zip",
        },
    )
    assert response.status_code in (401, 403)


def test_create_ml_task_invalid_model(client, auth_user):
    response = client.post(
        "/api/predict/",
        headers=auth_user["headers"],
        json={"ml_model_id": 93, "image_url": "patient.zip"},
    )
    assert response.status_code == 404


def test_create_ml_task_invalid_age(client, ml_model, auth_user):
    response = client.post(
        "/api/predict/",
        headers=auth_user["headers"],
        json={
            "ml_model_id": ml_model.id,
            "image_url": "patient.zip",
            "patient_age": -5,
        },
    )
    assert response.status_code == 422
