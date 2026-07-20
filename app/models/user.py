from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
import enum
import re
from pydantic import field_validator

if TYPE_CHECKING:
    from .ml_task import MLTask

EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
EMAIL_ERROR = "Укажите корректный email. Пример: name@example.com"


def validate_email(email: str) -> str:
    value = (email or "").strip()
    if not EMAIL_PATTERN.match(value):
        raise ValueError(EMAIL_ERROR)
    return value


class UserRole(str, enum.Enum):
    CLIENT = "client"
    ADMIN = "admin"


class User(SQLModel, table=True):
    """Пользователь системы."""

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True)
    email: str = Field(
        ...,
        unique=True,
        index=True,
        min_length=5,
        max_length=255
    )
    password: str = Field(..., min_length=4)
    role: UserRole = Field(default=UserRole.CLIENT)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    tasks: List["MLTask"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "selectin"
        }
    )

    def __str__(self) -> str:
        return f"Id: {self.id}. Username: {self.username}. Email: {self.email}"

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        return validate_email(value)

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
