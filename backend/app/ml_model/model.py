"""
Улучшенная модель аутентификации по клавиатурному почерку.
Конвейер: RobustScaler -> Outlier Filtering -> PCA (Whiten) -> Mahalanobis (Ledoit-Wolf).
"""

import pickle
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, EllipticEnvelope

from app.settings import env_settings


class KeystrokeModel:
    """
    Улучшенная модель PCA + Mahalanobis.
    Использует робастные методы для защиты от аномалий в поведении пользователя.
    """

    def __init__(self, confidence: float = 0.95, pca_max_components: int = 40):
        """
        :param confidence: Уровень доверия для теоретического порога (0.95 = 95%).
        :param pca_max_components: Максимальное количество компонент PCA.
        """
        self.confidence = confidence
        self.pca_max_components = pca_max_components

        # Используем RobustScaler вместо StandardScaler, чтобы случайные задержки 
        # (выбросы) не искажали масштаб всех признаков.
        self.scaler = RobustScaler()
        self.pca: Optional[PCA] = None

        self.is_trained: bool = False
        self.threshold: Optional[float] = None
        self.feature_names: Optional[List[str]] = None

        self._mu: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._keep_mask: Optional[np.ndarray] = None

    def fit(self, feature_vectors: List[List[float]], feature_names: List[str]) -> None:
        if not feature_vectors or len(feature_vectors) < 10:
            raise ValueError(f"Нужно минимум 10 попыток, получено {len(feature_vectors)}")

        X = np.array(feature_vectors, dtype=float)

        # 1. Более строгий фильтр признаков
        # Убираем признаки, где почти нет изменений (std < 0.0001)
        stds = X.std(axis=0)
        self._keep_mask = stds > 1e-4 
        X = X[:, self._keep_mask]
        self.feature_names = list(np.array(feature_names)[self._keep_mask])

        # 2. Робастное масштабирование
        X_scaled = self.scaler.fit_transform(X)

        n_samples, n_features = X_scaled.shape

        # 3. Динамический расчет компонент PCA (Исправлено!)
        # Количество компонент не может быть больше n_samples - 1.
        # Чтобы матрица была Full Rank, берем еще меньше.
        suggested_components = min(n_samples - 2, self.pca_max_components, n_features)
        n_components = max(2, suggested_components)
        
        self.pca = PCA(n_components=n_components, whiten=True, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # 4. Ledoit-Wolf с защитой
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X_pca)
        
        self._mu = X_pca.mean(axis=0)
        
        # Добавляем "малое смещение" (Ridge regularization), чтобы матрица всегда была обратимой
        cov = lw.covariance_
        eye = np.eye(n_components)
        self._cov_inv = np.linalg.inv(cov + 1e-6 * eye)

        # 5. Порог
        chi2_val = float(np.sqrt(stats.chi2.ppf(self.confidence, df=n_components)))
        train_dists = np.array([self._mahalanobis_raw(x) for x in X_pca])
        empirical_val = float(np.percentile(train_dists, 90)) * 2.0

        self.threshold = max(chi2_val, empirical_val)
        self.is_trained = True

    def _mahalanobis_raw(self, x: np.ndarray) -> float:
        """Внутренняя функция расчета расстояния Махаланобиса."""
        delta = x - self._mu
        return float(np.sqrt(delta @ self._cov_inv @ delta))

    def _align_and_project(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Преобразование входящего словаря в вектор признаков PCA."""
        if self.feature_names is None or self.pca is None:
            raise RuntimeError("Модель не обучена.")
        
        vec = np.array(
            [float(feature_dict.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        x_sc = self.scaler.transform(vec.reshape(1, -1))
        return self.pca.transform(x_sc)[0]

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Проверка входящей попытки.
        """
        if not self.is_trained:
            raise RuntimeError("Модель не обучена.")

        try:
            x_pca = self._align_and_project(feature_dict)
            score = self._mahalanobis_raw(x_pca)
        except Exception as e:
            # Если возникла ошибка (например, новые признаки), возвращаем отказ
            return {"score": 999.0, "threshold": self.threshold, "accepted": False, "confidence": 0.0}

        # Confidence: 1.0 (идеально) -> 0.0 (на границе порога)
        conf = round(max(0.0, min(1.0, 1.0 - (score / self.threshold))), 3)
        
        # Интеграция с настройками системы
        min_conf = getattr(env_settings, "ThresholdOfConfidence", 50)
        
        accepted = (score <= self.threshold and (conf * 100) >= min_conf)

        return {
            "score":      float(score),
            "threshold":  float(self.threshold),
            "accepted":   bool(accepted),
            "confidence": float(conf),
        }

    def save_to_bytes(self) -> bytes:
        """Сериализация модели в байты для хранения в БД."""
        return pickle.dumps(self)

    def save(self, path: str) -> None:
        """Сохранение модели в файл."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "KeystrokeModel":
        """Загрузка модели из файла."""
        with open(path, "rb") as f:
            return pickle.load(f)


def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Извлечение векторов признаков из JSON. 
    Гарантирует прямоугольную матрицу (union всех имен признаков).
    """
    flat_list: List[Dict[str, float]] = []

    attempts = parsed_json.get("attempts", [])
    for attempt in attempts:
        # Проверяем наличие признаков
        feats = attempt.get("features", {})
        flat = feats.get("flat_features")
        
        # Берем только валидные или имеющиеся попытки
        if flat:
            flat_list.append(flat)

    if not flat_list:
        return [], []

    # Собираем все уникальные ключи признаков
    all_names = sorted(set().union(*(f.keys() for f in flat_list)))
    
    # Строим векторы
    vectors = []
    for f in flat_list:
        vectors.append([f.get(name, 0.0) for name in all_names])

    return vectors, all_names

# ---------------------------------------------------------------------------
# Helper for training data extraction
# ---------------------------------------------------------------------------

def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Extract (vectors, feature_names) from the output of transform_payload().

    Different attempts may produce different n-gram features (digraphs/trigraphs
    depend on the exact keys pressed, including typos and corrections).  Collecting
    feature_vector from the first attempt and reusing its feature_names for all
    others results in vectors of different lengths -> inhomogeneous numpy array.

    Fix: collect flat_features dicts from every attempt, compute the *union* of all
    feature names, then rebuild each vector by looking up names in the dict (missing
    features -> 0.0).  The resulting matrix is always rectangular.
    """
    flat_list: List[Dict[str, float]] = []

    for attempt in parsed_json.get("attempts", []):
        flat = attempt.get("features", {}).get("flat_features")
        if flat:
            flat_list.append(flat)

    if not flat_list:
        return [], None

    all_names = sorted(set().union(*(f.keys() for f in flat_list)))
    vectors   = [[f.get(name, 0.0) for name in all_names] for f in flat_list]

    return vectors, all_names
