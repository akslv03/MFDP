from database.config import get_settings
from models.user import User
from sqlmodel import select


def test_signup(client, session):
    response = client.post(
        "/auth/signup",
        json={
            "username": "tester",
            "email": "test@test.ru",
            "password": "password123",
        },
    )

    assert response.status_code == 201
    assert response.json() == {"message": "Пользователь успешно зарегистрирован"}

    user = session.exec(select(User).where(User.email == "test@test.ru")).first()
    assert user is not None
    assert user.username == "tester"


def test_login_and_get_token(client):
    client.post(
        "/auth/signup",
        json={
            "username": "tester",
            "email": "test@test.ru",
            "password": "password123",
        },
    )

    response = client.post(
        "/auth/token",
        data={
            "username": "test@test.ru",
            "password": "password123",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    cookie_name = get_settings().COOKIE_NAME
    assert cookie_name in payload
    assert payload["token_type"] == "bearer"
    assert "access_token" in payload
    assert payload["email"] == "test@test.ru"
    assert payload["user_id"] is not None


def test_signup_short_password(client):
    response = client.post(
        "/auth/signup",
        json={
            "username": "tester",
            "email": "shortpass@test.ru",
            "password": "123",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Пароль должен содержать не менее 4 символов" in detail


def test_signup_invalid_email(client):
    response = client.post(
        "/auth/signup",
        json={
            "username": "tester",
            "email": "not-an-email",
            "password": "password123",
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "@" in detail or "email" in detail.lower()


def test_me_endpoint(client):
    client.post(
        "/auth/signup",
        json={
            "username": "me_user",
            "email": "me@test.ru",
            "password": "password123",
        },
    )
    token_resp = client.post(
        "/auth/token",
        data={"username": "me@test.ru", "password": "password123"},
    )
    token = token_resp.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "me@test.ru"
