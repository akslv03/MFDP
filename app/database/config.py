from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: Optional[str] = None
    DB_PASS: Optional[str] = None
    DB_NAME: Optional[str] = None
    COOKIE_NAME: Optional[str] = None
    SECRET_KEY: Optional[str] = None

    APP_NAME: Optional[str] = None
    APP_DESCRIPTION: Optional[str] = None
    DEBUG: Optional[bool] = None
    API_VERSION: Optional[str] = None

    RABBITMQ_HOST: Optional[str] = None
    RABBITMQ_PORT: Optional[int] = None
    RABBITMQ_USER: Optional[str] = None
    RABBITMQ_PASS: Optional[str] = None
    RABBITMQ_QUEUE_NAME: Optional[str] = None

    ADMIN_EMAIL: str = "admin@mri.local"
    ADMIN_USERNAME: str = "AdminUser"
    ADMIN_PASSWORD: str = Field(..., min_length=8)
    DEMO_EMAIL: str = "demo@client.com"
    DEMO_USERNAME: str = "DemoClient"
    DEMO_PASSWORD: str = Field(..., min_length=8)

    @property
    def DATABASE_URL_psycopg(self):
        return f'postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'

    model_config = SettingsConfigDict(
        env_file=(".env", "app/.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def validate(self) -> None:
        """Проверяет, что заданы обязательные параметры БД."""
        if not all([self.DB_HOST, self.DB_USER, self.DB_PASS, self.DB_NAME]):
            raise ValueError("Не заданы обязательные параметры подключения к БД")
        if not self.ADMIN_PASSWORD or len(self.ADMIN_PASSWORD) < 8:
            raise ValueError("ADMIN_PASSWORD не задан или короче 8 символов")
        if not self.DEMO_PASSWORD or len(self.DEMO_PASSWORD) < 8:
            raise ValueError("DEMO_PASSWORD не задан или короче 8 символов")

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
