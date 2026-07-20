import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine
from unittest.mock import patch

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASS", "postgres")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("COOKIE_NAME", "MRI_ACCESS_TOKEN")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("APP_NAME", "MRI Tumor Segmentation API")
os.environ.setdefault("APP_DESCRIPTION", "test")
os.environ.setdefault("DEBUG", "False")
os.environ.setdefault("API_VERSION", "1.0")
os.environ.setdefault("RABBITMQ_HOST", "localhost")
os.environ.setdefault("RABBITMQ_PORT", "5672")
os.environ.setdefault("RABBITMQ_USER", "guest")
os.environ.setdefault("RABBITMQ_PASS", "guest")
os.environ.setdefault("RABBITMQ_QUEUE_NAME", "ml_task_queue")

from api import app
from database.database import get_session
from models.ml_model import MLModel


@pytest.fixture(name="session")
def session_fixture(tmp_path):
    db_path = tmp_path / "testing.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        yield session

    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_rabbitmq():
    with patch("routes.predict.send_task"):
        yield


@pytest.fixture
def ml_model(session: Session):
    model = MLModel(
        id=1,
        name="test_model",
        description="Test model",
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


@pytest.fixture
def auth_user(client, session):
    client.post(
        "/auth/signup",
        json={"username": "doc", "email": "doc@test.ru", "password": "password123"},
    )
    token_resp = client.post(
        "/auth/token",
        data={"username": "doc@test.ru", "password": "password123"},
    )
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]
    user_id = token_resp.json()["user_id"]
    return {
        "headers": {"Authorization": f"Bearer {token}"},
        "user_id": user_id,
        "email": "doc@test.ru",
    }
