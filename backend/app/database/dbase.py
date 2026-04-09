import pickle
from typing import Optional, AsyncGenerator

from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Предполагаем, что настройки импортируются отсюда, как в твоем примере
# Если структуры папок нет, можно заменить на прямые строки
try:
    from app.settings import env_settings
except ImportError:
    # Заглушка для примера, если файл настроек не найден
    class Settings:
        DATABASE_PREFIX = "sqlite+aiosqlite:///"
        DATABASE_USER = ""
        POSTGRES_PASSWORD = ""
        DATABASE_HOST = ""
        DATABASE_PORT = None
        DATABASE_NAME = "keystroke_data.db"
    env_settings = Settings()

class DatabaseUrlBuilder():
    def __init__(self) -> None:
        self.preffix: Optional[str] = None
        self.user: Optional[str] = None
        self.password: Optional[str] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self.name: Optional[str] = None

    def WithPrefix(self, preffix: str):
        self.preffix = preffix
        return self

    def WithUser(self, user: str):
        self.user = user
        return self

    def WithPassword(self, password: Optional[str]):
        self.password = password
        return self

    def WithHost(self, host: Optional[str]):
        self.host = host
        return self

    def WithPort(self, port: Optional[int]):
        self.port = port
        return self
    
    def WithName(self, name: str):
        self.name = name
        return self

    def Build(self) -> str:
        # Упрощенная логика сборки для разных типов БД
        if "sqlite" in (self.preffix or ""):
            return f"{self.preffix}{self.name}"
            
        return (
            f"{self.preffix}{self.user}"
            f"{':' + self.password if self.password else ''}"
            f"@{self.host if self.host else ''}"
            f"{':' + str(self.port) if self.port else ''}"
            f"/{self.name}"
        )

class Base(DeclarativeBase):
    pass

class User(Base):
    """
    Единственная таблица для хранения пользователей и их обученных моделей.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    login = Column(String, unique=True, index=True, nullable=False)
    # Поле для хранения дампа модели (вариант 2: BLOB / BYTEA)
    model_data = Column(LargeBinary, nullable=True)

    def set_model(self, model_obj):
        """Сериализует объект модели в байты."""
        self.model_data = pickle.dumps(model_obj)

    def get_model(self):
        """Десериализует байты обратно в объект модели."""
        if self.model_data:
            return pickle.loads(self.model_data)
        return None

class Database():
    def __init__(self, url: Optional[str] = None):
        self.db_url: str = url if url else \
            DatabaseUrlBuilder() \
            .WithPrefix(env_settings.DATABASE_PREFIX) \
            .WithUser(env_settings.DATABASE_USER) \
            .WithPassword(env_settings.POSTGRES_PASSWORD) \
            .WithHost(env_settings.DATABASE_HOST) \
            .WithPort(env_settings.DATABASE_PORT) \
            .WithName(env_settings.DATABASE_NAME) \
            .Build()

        self.async_engine: AsyncEngine = create_async_engine(self.db_url, echo=True)

        self.AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
        )

    async def migrate_models(self) -> None:
        """Создает таблицы в базе данных."""
        async with self.async_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    
# Инициализация объекта БД
db_instance: Database = Database()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Зависимость для получения сессии (удобно для FastAPI)."""
    async with db_instance.AsyncSessionLocal() as session:
        yield session