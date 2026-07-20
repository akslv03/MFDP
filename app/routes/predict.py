import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional
from auth.authenticate import authenticate_cookie
from database.database import get_session
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from models.ml_model import MLModel
from models.ml_task import MLTask, TaskStatus, DoctorReview
from pydantic import BaseModel, Field, field_validator
from services.crud import ml_task as TaskService
from services.crud import user as UserService
from services.rm.rm import send_task
from sqlmodel import Session

logger = logging.getLogger(__name__)

predict_route = APIRouter()

SUPPORTED_EXTS = {".zip", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))


def _uploads_dir() -> str:
    configured = os.getenv("UPLOAD_DIR")
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured

    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def _is_supported_ext(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTS


def _require_supported_filename(filename: Optional[str]) -> str:
    name = os.path.basename(filename or "")
    if not _is_supported_ext(name):
        raise HTTPException(
            status_code=400,
            detail="Загрузите ZIP со срезами МРТ или изображение (png/tif/jpg/bmp)",
        )
    return name


def _save_upload(upload: UploadFile, dest_path: str) -> None:
    written = 0
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                buffer.close()
                os.remove(dest_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл больше лимита ({MAX_UPLOAD_BYTES // (1024 * 1024)} МБ)",
                )
            buffer.write(chunk)


def _parse_age(raw: Optional[str | int]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        age = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Возраст должен быть целым числом") from exc
    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Возраст должен быть в диапазоне 0–120")
    return age


def _parse_gender(raw: Optional[str]) -> Optional[str]:
    if raw is None or raw == "" or raw == "unknown":
        return None
    value = str(raw).strip()
    if value not in {"1", "2"}:
        raise HTTPException(
            status_code=400,
            detail="Пол: допустимы 1, 2",
        )
    return value


def _current_user(user_email: str, session: Session):
    user = UserService.get_user_by_email(user_email, session)
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return user


def _enqueue_task(
    *,
    session: Session,
    user_id: int,
    ml_model_id: int,
    file_path: str,
    age: Optional[int],
    gender: Optional[str],
) -> MLTask:
    model = session.get(MLModel, ml_model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Модель не найдена")

    new_task = MLTask(
        user_id=user_id,
        ml_model_id=ml_model_id,
        image_url=file_path,
        patient_age=age,
        patient_gender=gender,
        status=TaskStatus.CREATED,
    )
    saved_task = TaskService.create_task(new_task, session)

    message = {
        "task_id": saved_task.id,
        "features": {
            "image_path": file_path,
            "patient_age": age,
            "patient_gender": gender,
        },
        "model": model.name,
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        send_task(message)
    except Exception as exc:
        logger.error("Failed to publish task %s: %s", saved_task.id, exc)
        saved_task.status = TaskStatus.FAILED
        saved_task.error_message = "Не удалось поставить задачу в очередь"
        session.add(saved_task)
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Очередь задач недоступна, попробуйте позже",
        ) from exc
    return saved_task


@predict_route.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Загрузка ZIP с МРТ и создание задачи сегментации",
)
async def create_segmentation_task_upload(
    ml_model_id: int = Form(1),
    patient_age: Optional[str] = Form(None),
    patient_gender: Optional[str] = Form(None),
    image: UploadFile = File(...),
    user_email: str = Depends(authenticate_cookie),
    session: Session = Depends(get_session),
):
    user = _current_user(user_email, session)
    age = _parse_age(patient_age)
    gender = _parse_gender(patient_gender)
    original_name = _require_supported_filename(image.filename)

    upload_dir = _uploads_dir()
    unique_name = f"{user.id}_{uuid.uuid4().hex[:10]}_{original_name}"
    file_path = os.path.join(upload_dir, unique_name)

    _save_upload(image, file_path)

    saved_task = _enqueue_task(
        session=session,
        user_id=user.id,
        ml_model_id=ml_model_id,
        file_path=file_path,
        age=age,
        gender=gender,
    )

    return {
        "task_id": saved_task.id,
        "status": saved_task.status,
        "image_url": file_path,
        "patient_age": age,
        "patient_gender": gender,
    }


class PredictRequest(BaseModel):
    ml_model_id: int = 1
    image_url: str
    patient_age: Optional[int] = Field(default=None, ge=0, le=120)
    patient_gender: Optional[str] = None

    @field_validator("image_url")
    @classmethod
    def supported_only(cls, value: str) -> str:
        if not _is_supported_ext(str(value)):
            raise ValueError(
                "Нужен путь к ZIP со срезами МРТ или к изображению (png/tif/jpg/bmp)"
            )
        return value


@predict_route.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Создать задачу сегментации по пути к ZIP",
)
async def create_segmentation_task(
    request: PredictRequest,
    user_email: str = Depends(authenticate_cookie),
    session: Session = Depends(get_session),
):
    try:
        user = _current_user(user_email, session)
        gender = _parse_gender(request.patient_gender)
        saved_task = _enqueue_task(
            session=session,
            user_id=user.id,
            ml_model_id=request.ml_model_id,
            file_path=request.image_url,
            age=request.patient_age,
            gender=gender,
        )
        return {"task_id": saved_task.id, "status": saved_task.status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error publishing task to RabbitMQ: %s", e)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@predict_route.get(
    "/tasks/mine",
    response_model=List[MLTask],
    summary="Мои задачи сегментации",
)
async def list_my_tasks(
    user_email: str = Depends(authenticate_cookie),
    session: Session = Depends(get_session),
) -> List[MLTask]:
    user = _current_user(user_email, session)
    return TaskService.get_user_tasks(user.id, session)


class ReviewRequest(BaseModel):
    decision: DoctorReview

    @field_validator("decision")
    @classmethod
    def only_accept_or_reject(cls, value: DoctorReview) -> DoctorReview:
        if value not in {DoctorReview.ACCEPTED, DoctorReview.REJECTED}:
            raise ValueError("Допустимо принято или отклонено")
        return value


@predict_route.post(
    "/{task_id}/review",
    response_model=MLTask,
    summary="Принять или отклонить результат сегментации",
)
async def set_doctor_review(
    task_id: int,
    request: ReviewRequest,
    user_email: str = Depends(authenticate_cookie),
    session: Session = Depends(get_session),
) -> MLTask:
    user = _current_user(user_email, session)
    task = session.get(MLTask, task_id)
    if not task or task.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Задача не найдена",
        )
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Оценку можно поставить только для завершенной задачи",
        )
    task.doctor_review = request.decision
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


@predict_route.get(
    "/{task_id}",
    response_model=MLTask,
    summary="Статус задачи по id",
)
async def get_task_status(
    task_id: int,
    user_email: str = Depends(authenticate_cookie),
    session: Session = Depends(get_session),
) -> MLTask:
    user = _current_user(user_email, session)
    task = session.get(MLTask, task_id)
    if not task or task.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Задача не найдена",
        )
    return task
