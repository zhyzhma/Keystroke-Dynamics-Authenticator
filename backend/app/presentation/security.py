from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.dbase import get_db, UserDeviceModel
from app.feature_engineering.engineering import transform_payload
from app.ml_model.model import KeystrokeModel, extract_training_data

router = APIRouter(prefix="/security", tags=["security"])

# --- Входные Pydantic модели ---

class RawEvent(BaseModel):
    """
    Одно событие клавиатуры/ввода из фронтенда.

    Поля key и code обязательны только для keydown/keyup; остальные типы событий
    (focus, blur, paste, input, beforeinput, compositionstart/update/end) их не имеют.
    Дополнительные поля (value, caretStart, caretEnd, inputType, data, location и др.)
    принимаются и передаются дальше в feature engineering через extra='allow'.
    """
    eventType: Optional[str] = Field(None, alias="type")
    key: Optional[str] = None
    code: Optional[str] = None
    t: float
    repeat: Optional[bool] = False

    class Config:
        populate_by_name = True
        extra = "allow"

class KeystrokeAttempt(BaseModel):
    attemptId: Optional[str] = "unnamed_attempt"
    events: List[RawEvent]

class EnrollRequest(BaseModel):
    login: str
    phrase: str
    device_type: str = Field("desktop", description="desktop или mobile")
    attempts: List[KeystrokeAttempt]

class VerifyRequest(BaseModel):
    login: str
    phrase: str
    device_type: str = "desktop"
    attempt: KeystrokeAttempt

# --- Выходные Pydantic модели ---

class EnrollResponse(BaseModel):
    status: str = "success"
    message: str
    attempts_count: int

class VerifyResponse(BaseModel):
    accepted: bool
    score: float
    threshold: float
    confidence: float
    message: str

# --- Эндпоинты ---

@router.post("/enroll", response_model=EnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_user(
        payload: EnrollRequest,
        db: AsyncSession = Depends(get_db)
    ) -> EnrollResponse:
    """
    Регистрация клавиатурного почерка.
    Принимает массив попыток, обучает One-Class SVM и сохраняет веса в БД.
    """
    raw_data = payload.model_dump(by_alias=True)
    processed_data = transform_payload(raw_data)
    vectors, names = extract_training_data(processed_data)

    total = len(payload.attempts)
    valid = len(vectors)

    if valid < 10:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Недостаточно корректных попыток: {valid}/{total}. "
                "Нужно минимум 10 (рекомендуется 30+). "
                "Убедитесь, что текст вводится точно без опечаток."
            ),
        )

    model = KeystrokeModel()
    model.fit(vectors, names, phrase=payload.phrase)

    # BUG FIX: enroll now uses UserDeviceModel (same table as verify),
    # so the stored model is actually found during verification.
    # Previously enroll wrote to User.model_data while verify read from
    # UserDeviceModel.model_blob — two separate tables, so verify always 404'd.
    result = await db.execute(
        select(UserDeviceModel).filter_by(login=payload.login, device_type=payload.device_type)
    )
    record = result.scalar_one_or_none()

    if not record:
        record = UserDeviceModel(login=payload.login, device_type=payload.device_type)
        db.add(record)

    record.set_model(model)
    await db.commit()
    
    return EnrollResponse(
        message=f"Модель успешно обучена и сохранена ({valid}/{total} попыток корректны)",
        attempts_count=valid,
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_user(
        payload: VerifyRequest,
        db: AsyncSession = Depends(get_db)
    ) -> VerifyResponse:
    """
    Верификация пользователя по одной попытке ввода.
    """
    result = await db.execute(
        select(UserDeviceModel).filter_by(login=payload.login, device_type=payload.device_type)
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    
    # BUG FIX: was checking record.model_data — field doesn't exist on UserDeviceModel.
    # Correct field is model_blob (defined in dbase.py).
    if not record.model_blob:
        raise HTTPException(status_code=400, detail="Для пользователя не обучена модель")

    model: KeystrokeModel = record.get_model()

    raw_data = {
        "userId": payload.login,
        "phrase": payload.phrase,
        "attempts": [payload.attempt.model_dump(by_alias=True)]
    }
    processed_data = transform_payload(raw_data)

    if not processed_data["attempts"]:
        raise HTTPException(status_code=400, detail="Не удалось обработать данные ввода")

    attempt_feats = processed_data["attempts"][0]["features"]

    if not attempt_feats.get("valid"):
        return VerifyResponse(
            accepted=False,
            score=0.0,
            threshold=float(model.threshold or 0.0),
            confidence=0.0,
            message="Введённый текст не совпадает с фразой — доступ запрещён",
        )

    prediction = model.predict(attempt_feats["flat_features"])

    return VerifyResponse(
        accepted=prediction["accepted"],
        score=prediction["score"],
        threshold=prediction["threshold"],
        confidence=prediction["confidence"],
        message="Доступ разрешён" if prediction["accepted"] else "Доступ запрещён: почерк не совпадает",
    )
