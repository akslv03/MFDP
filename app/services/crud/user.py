from models.user import User
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from typing import Optional

def get_user_by_email(email: str, session: Session) -> Optional[User]:
    """Ищет пользователя по email."""
    statement = select(User).where(User.email == email).options(
        selectinload(User.tasks)
    )
    return session.exec(statement).first()

def create_user(user: User, session: Session) -> User:
    """Сохраняет нового пользователя в базу данных."""
    try:
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    except Exception:
        session.rollback()
        raise
