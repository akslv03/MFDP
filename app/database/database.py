from sqlmodel import SQLModel, Session, create_engine, select
from auth.hash_password import HashPassword
from models.ml_model import MLModel
from models.user import User, UserRole
from .config import get_settings


def get_database_engine():
    settings = get_settings()
    engine = create_engine(
        url=settings.DATABASE_URL_psycopg,
        echo=settings.DEBUG,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    return engine


engine = get_database_engine()


def get_session():
    with Session(engine) as session:
        yield session


def init_db(drop_all: bool = False) -> None:
    try:
        settings = get_settings()
        engine = get_database_engine()
        if drop_all:
            SQLModel.metadata.drop_all(engine)

        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            hasher = HashPassword()

            default_model = session.exec(
                select(MLModel).where(MLModel.name == "brain_mri_unet")
            ).first()
            if not default_model:
                default_model = MLModel(
                    name="brain_mri_unet",
                    description=(
                        "Сегментация опухоли на МРТ головного мозга: "
                        "UNet + SegFormer mit_b2"
                    ),
                )
                session.add(default_model)
                print("Создана ML-модель: brain_mri_unet")

            admin = session.exec(
                select(User).where(User.email == settings.ADMIN_EMAIL)
            ).first()
            if not admin:
                admin = User(
                    username=settings.ADMIN_USERNAME,
                    email=settings.ADMIN_EMAIL,
                    password=hasher.create_hash(settings.ADMIN_PASSWORD),
                    role=UserRole.ADMIN,
                )
                session.add(admin)
                print(f"Создан демо-администратор: {settings.ADMIN_EMAIL}")

            demo_user = session.exec(
                select(User).where(User.email == settings.DEMO_EMAIL)
            ).first()
            if not demo_user:
                demo_user = User(
                    username=settings.DEMO_USERNAME,
                    email=settings.DEMO_EMAIL,
                    password=hasher.create_hash(settings.DEMO_PASSWORD),
                    role=UserRole.CLIENT,
                )
                session.add(demo_user)
                print(f"Создан демо-пользователь: {settings.DEMO_EMAIL}")

            session.commit()
            print("База данных успешно инициализирована.")

    except Exception as e:
        print(f"Ошибка при инициализации БД: {e}")
        raise
