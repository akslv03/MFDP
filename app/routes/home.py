import os
from typing import Dict
from auth.authenticate import authenticate_cookie
from database.database import get_session
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from services.crud import user as UserService
from sqlmodel import Session

home_route = APIRouter()


@home_route.get("/")
async def index():
    streamlit_url = os.getenv("STREAMLIT_PUBLIC_URL", "http://localhost:8501")
    return RedirectResponse(url=streamlit_url)


@home_route.get("/uploads/{filename}")
async def serve_upload(
    filename: str,
    user_email: str = Depends(authenticate_cookie),
    session: Session = Depends(get_session),
):
    user = UserService.get_user_by_email(user_email, session)
    if user is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    name = os.path.basename(filename)
    if name != filename or not name.startswith(f"{user.id}_"):
        raise HTTPException(status_code=404, detail="Файл не найден")

    candidates = []
    upload_dir = os.getenv("UPLOAD_DIR")
    if upload_dir:
        candidates.append(os.path.join(upload_dir, name))
    candidates.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads", name))
    )
    candidates.append(os.path.join("/app/uploads", name))

    for file_path in candidates:
        if os.path.exists(file_path):
            return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Файл не найден")


@home_route.get(
    "/health",
    response_model=Dict[str, str],
    summary="Health check endpoint",
    description="Returns service health status",
)
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}
