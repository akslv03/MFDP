import logging
from typing import Any, Dict
from auth.authenticate import authenticate_cookie
from auth.hash_password import HashPassword
from auth.jwt_handler import create_access_token
from database.config import get_settings
from database.database import get_session
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from models.user import User, validate_email
from pydantic import BaseModel, Field, field_validator
from services.crud import user as UserService

logger = logging.getLogger(__name__)

auth_route = APIRouter()
hash_password = HashPassword()
settings = get_settings()

PASSWORD_MIN_LENGTH = 4


def _validate_password_value(password: str) -> str:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Пароль должен содержать не менее {PASSWORD_MIN_LENGTH} символов"
        )
    return password


def _validate_username_value(username: str) -> str:
    value = (username or "").strip()
    if not value:
        raise ValueError("Укажите имя пользователя")
    if len(value) < 2:
        raise ValueError("Имя пользователя должно содержать не менее 2 символов")
    return value


class UserSignup(BaseModel):
    username: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

    @field_validator("username")
    @classmethod
    def username_ok(cls, value: str) -> str:
        return _validate_username_value(value)

    @field_validator("email")
    @classmethod
    def email_ok(cls, value: str) -> str:
        return validate_email(value)

    @field_validator("password")
    @classmethod
    def password_ok(cls, value: str) -> str:
        return _validate_password_value(value)


@auth_route.post("/token")
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session=Depends(get_session),
) -> Dict[str, Any]:
    user = UserService.get_user_by_email(form_data.username, session)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    if not hash_password.verify_hash(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
        )

    access_token = create_access_token(user.email)

    response.set_cookie(
        key=settings.COOKIE_NAME,
        value=f"Bearer {access_token}",
        httponly=True,
    )

    return {
        settings.COOKIE_NAME: access_token,
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "username": user.username,
    }


@auth_route.post(
    "/signup",
    response_model=Dict[str, str],
    status_code=status.HTTP_201_CREATED,
    summary="User registration",
    description="Register a new user with email and password",
)
async def signup(data: UserSignup, session=Depends(get_session)) -> Dict[str, str]:
    """Регистрация нового пользователя."""
    try:
        if UserService.get_user_by_email(data.email, session):
            logger.warning(f"Signup attempt with existing email: {data.email}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Пользователь с таким email уже существует",
            )
        hasher = HashPassword()
        hashed_password = hasher.create_hash(data.password)

        new_user = User(username=data.username, email=data.email, password=hashed_password)
        UserService.create_user(new_user, session)

        logger.info(f"New user registered: {data.email}")
        return {"message": "Пользователь успешно зарегистрирован"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при создании пользователя",
        )


@auth_route.get("/me", summary="Current user profile")
async def get_me(
    user_email: str = Depends(authenticate_cookie),
    session=Depends(get_session),
):
    user = UserService.get_user_by_email(user_email, session)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": str(user.role.value if hasattr(user.role, "value") else user.role),
    }
