import time
from datetime import datetime
from fastapi import HTTPException, status
from jose import jwt, JWTError

from database.config import get_settings

settings = get_settings()
SECRET_KEY = settings.SECRET_KEY


def create_access_token(user: str) -> str:
    """
    Создает JWT-токен для переданного email.
    Токен живет 1 час.
    """
    payload = {
        "user": user,
        "expires": time.time() + 3600,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


def verify_access_token(token: str) -> dict:
    """Проверяет JWT и срок действия, возвращает payload."""
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        expire = data.get("expires")
        if expire is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Токен доступа не передан",
            )
        if datetime.utcnow() > datetime.utcfromtimestamp(expire):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Срок действия токена истек",
            )
        return data
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Недействительный токен",
        )
