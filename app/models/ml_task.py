from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, TYPE_CHECKING
from datetime import datetime
import enum

if TYPE_CHECKING:
    from .user import User
    from .ml_model import MLModel

class TaskStatus(str, enum.Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DoctorReview(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class MLTask(SQLModel, table=True):
    """Задача сегментации опухоли."""

    __tablename__ = "ml_task"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    ml_model_id: int = Field(foreign_key="ml_model.id")
    image_url: str
    patient_age: Optional[int] = Field(default=None)
    patient_gender: Optional[str] = Field(default=None)
    status: TaskStatus = Field(default=TaskStatus.CREATED)
    display_image_path: Optional[str] = Field(default=None)
    result_mask_path: Optional[str] = Field(default=None)
    overlay_image_path: Optional[str] = Field(default=None)
    similarity_cases: Optional[str] = Field(default=None)
    slice_gallery: Optional[str] = Field(default=None)
    doctor_review: DoctorReview = Field(default=DoctorReview.PENDING)
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: "User" = Relationship(back_populates="tasks")
    ml_model: "MLModel" = Relationship(back_populates="tasks")

    def __str__(self) -> str:
        return f"Task Id: {self.id}. Status: {self.status.value}"

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
