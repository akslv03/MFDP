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
                select(User).where(User.email == "admin@mri.local")
            ).first()
            if not admin:
                admin = User(
                    username="AdminUser",
                    email="admin@mri.local",
                    password=hasher.create_hash("secure_admin_pass"),
                    role=UserRole.ADMIN,
                )
                session.add(admin)
                print("Создан демо-администратор.")

            demo_user = session.exec(
                select(User).where(User.email == "demo@client.com")
            ).first()
            if not demo_user:
                demo_user = User(
                    username="DemoClient",
                    email="demo@client.com",
                    password=hasher.create_hash("demo_password"),
                    role=UserRole.CLIENT,
                )
                session.add(demo_user)
                print("Создан демо-пользователь: demo@client.com")

            session.commit()
            print("База данных успешно инициализирована.")

    except Exception as e:
        print(f"Ошибка при инициализации БД: {e}")
        raise
