from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .ml_task import MLTask

class MLModel(SQLModel, table=True):
    """Доступная ML-модель сегментации."""
    __tablename__ = "ml_model"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str

    tasks: List["MLTask"] = Relationship(back_populates="ml_model")
    
    def __str__(self) -> str:
        return f"Model Id: {self.id}. Name: {self.name}"

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
