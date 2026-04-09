from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Импорты согласно структуре проекта
from app.database.dbase import get_db, User
from app.feature_engineering.engineering import transform_payload
from app.ml_model.model import KeystrokeModel, extract_training_data

router = APIRouter(prefix="/security", tags=["security"])

# --- Входные Pydantic модели ---

class RawEvent(BaseModel):
    eventType: Optional[str] = Field(None, alias="type", description="Тип события (keydown/keyup)")
    key: str = Field(..., description="Значение клавиши")
    code: str = Field(..., description="Физический код клавиши")
    t: float = Field(..., description="Таймштамп события в мс")
    repeat: Optional[bool] = Field(False, description="Является ли событие автоповтором")

    class Config:
        populate_by_name = True

class KeystrokeAttempt(BaseModel):
    attemptId: Optional[str] = Field("unnamed_attempt", description="ID попытки ввода")
    events: List[RawEvent] = Field(..., description="Список событий нажатий")

class EnrollRequest(BaseModel):
    login: str = Field(..., description="Логин пользователя")
    phrase: str = Field(..., description="Фраза, которую вводил пользователь")
    attempts: List[KeystrokeAttempt] = Field(..., description="Список из 30-40 попыток для обучения")

class VerifyRequest(BaseModel):
    login: str = Field(..., description="Логин пользователя")
    phrase: str = Field(..., description="Введенная фраза")
    attempt: KeystrokeAttempt = Field(..., description="Текущая попытка ввода для проверки")

# --- Выходные Pydantic модели (с описанием) ---

class EnrollResponse(BaseModel):
    status: str = Field("success", description="Статус операции")
    message: str = Field(..., description="Текстовое описание результата")
    attempts_count: int = Field(..., description="Количество успешно обработанных попыток")

class VerifyResponse(BaseModel):
    accepted: bool = Field(..., description="Вердикт: прошел ли пользователь проверку")
    score: float = Field(..., description="Сырой балл схожести от модели")
    threshold: float = Field(..., description="Порог отсечения для данной модели")
    confidence: float = Field(..., description="Уровень уверенности системы (от 0 до 1)")
    message: str = Field(..., description="Текстовое уведомление для пользователя")

# --- Эндпоинты ---

@router.post("/enroll", response_model=EnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_user(
        payload: EnrollRequest,
        db: AsyncSession = Depends(get_db)
    ) -> EnrollResponse:
    
    """
    Регистрация клавиатурного почерка. 
    Принимает массив попыток, обучает модель One-Class SVM и сохраняет веса в БД.
    """
    raw_data = payload.model_dump(by_alias=True)
    processed_data = transform_payload(raw_data)
    vectors, names = extract_training_data(processed_data)
    
    if len(vectors) < 5:
        raise HTTPException(
            status_code=400, 
            detail="Недостаточно попыток для обучения. Нужно минимум 5 (рекомендуется 30+)."
        )

    # Обучение
    model = KeystrokeModel()
    model.fit(vectors, names)

    # Сохранение в БД
    result = await db.execute(select(User).filter_by(login=payload.login))
    user = result.scalar_one_or_none()

    if not user:
        user = User(login=payload.login)
        db.add(user)
    
    user.set_model(model)
    await db.commit()
    
    return EnrollResponse(
        status="success",
        message="Модель успешно обучена и сохранена",
        attempts_count=len(vectors)
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_user(
        payload: VerifyRequest,
        db: AsyncSession = Depends(get_db)
    ) -> VerifyResponse:

    """
    Верификация пользователя по одной попытке ввода.
    Сравнивает текущий ритм с 'эталоном', хранящимся в БД.
    """
    result = await db.execute(select(User).filter_by(login=payload.login))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    if not user.model_data:
        raise HTTPException(status_code=400, detail="Для пользователя не обучена модель")

    # Восстановление модели из байтов
    model: KeystrokeModel = user.get_model()

    # Feature Engineering для текущей попытки
    raw_data = {
        "userId": payload.login,
        "phrase": payload.phrase,
        "attempts": [payload.attempt.model_dump(by_alias=True)]
    }
    processed_data = transform_payload(raw_data)
    
    if not processed_data["attempts"]:
        raise HTTPException(status_code=400, detail="Не удалось обработать данные ввода")
    
    current_features = processed_data["attempts"][0]["features"]["flat_features"]

    # Проверка
    prediction = model.predict(current_features)

    return VerifyResponse(
        accepted=prediction["accepted"],
        score=prediction["score"],
        threshold=prediction["threshold"],
        confidence=prediction["confidence"],
        message="Доступ разрешен" if prediction["accepted"] else "Доступ запрещен: почерк не совпадает"
    )