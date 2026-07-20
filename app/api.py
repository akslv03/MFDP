from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.home import home_route
from routes.auth import auth_route
from routes.cases import cases_route
from routes.predict import predict_route
from database.config import get_settings
from database.database import init_db
import uvicorn
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Инициализация базы данных...")
        init_db()
        logger.info("Успешный запуск приложения")
    except Exception as e:
        logger.error(f"Ошибка при запуске: {str(e)}")
        raise
    yield
    logger.info("Остановка приложения...")


def create_application() -> FastAPI:
    """Собирает FastAPI-приложение и подключает роуты."""

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(home_route, tags=['Home'])
    app.include_router(auth_route, prefix='/auth', tags=['Auth'])
    app.include_router(cases_route, prefix='/api/cases', tags=['Historical Cases'])
    app.include_router(predict_route, prefix='/api/predict', tags=['ML Predictions'])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        messages: list[str] = []
        for err in exc.errors():
            msg = err.get("msg", "Ошибка валидации")
            if isinstance(msg, str) and msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            messages.append(msg)
        detail = "; ".join(messages) if messages else "Ошибка валидации данных"
        return JSONResponse(status_code=422, content={"detail": detail})

    return app

app = create_application()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=8080,
        reload=True,
        log_level="info"
    )