import numpy as np
import pickle
from typing import List, Dict, Any
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class KeystrokeModel:
    def __init__(self, nu: float = 0.1, gamma: str = "scale"):
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=self.nu, gamma=self.gamma)
        self.is_trained = False
        self.threshold = None
        self.feature_names = None

    def fit(self, feature_vectors: List[List[float]], feature_names: List[str]):
        X = np.array(feature_vectors)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.feature_names = feature_names

        scores = self.model.decision_function(X_scaled)
        # Используем 5-й перцентиль вместо min для устойчивости к шуму
        self.threshold = float(np.percentile(scores, 5)) 
        self.is_trained = True

    def _align_vector(self, feature_dict: Dict[str, float]) -> List[float]:
        """Приводит входящий словарь к эталонному вектору признаков"""
        if not self.feature_names:
            raise RuntimeError("Model has no feature names stored.")
        # Если признака нет в текущей попытке, ставим 0.0
        return [float(feature_dict.get(name, 0.0)) for name in self.feature_names]

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        
        vector = self._align_vector(feature_dict)
        x = np.array(vector).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        score = float(self.model.decision_function(x_scaled)[0])

        return {
            "score": score,
            "threshold": self.threshold,
            "accepted": score >= self.threshold,
            "confidence": round(max(0, (score - self.threshold) / (abs(self.threshold) + 1e-6)), 2)
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "threshold": self.threshold,
                "feature_names": self.feature_names
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]
        self.feature_names = data["feature_names"]
        self.is_trained = True

def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Превращает обработанный JSON в формат, пригодный для обучения модели.
    Достает списки чисел (векторы) и названия признаков.
    """
    vectors = []
    feature_names = None

    # Проходим по всем попыткам в сессии обучения
    for attempt in parsed_json.get("attempts", []):
        features = attempt.get("features", {})
        
        # Берем уже готовый числовой вектор
        vector = features.get("feature_vector", [])
        if vector:
            vectors.append(vector)

        # Запоминаем имена признаков (они одинаковы для всех попыток в одной сессии)
        if feature_names is None:
            feature_names = features.get("feature_names", [])

    return vectors, feature_names