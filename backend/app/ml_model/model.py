import pickle
import numpy as np
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
        """Обучение на матрице признаков."""
        if not feature_vectors or len(feature_vectors) < 5:
            raise ValueError("Insufficient data for training")

        X = np.array(feature_vectors)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
        self.feature_names = feature_names

        scores = self.model.decision_function(X_scaled)
        self.threshold = float(np.percentile(scores, 5))
        
        self.is_trained = True

    def _align_vector(self, feature_dict: Dict[str, float]) -> List[float]:
        """
        Превращает словарь признаков в упорядоченный список,
        соответствующий порядку, запомненному при обучении.
        """
        if not self.feature_names:
            raise RuntimeError("Model has no feature names stored.")
        return [float(feature_dict.get(name, 0.0)) for name in self.feature_names]

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Верификация по словарю признаков."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        
        vector = self._align_vector(feature_dict)
        x = np.array(vector).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        score = float(self.model.decision_function(x_scaled)[0])

        confidence = round(max(0, (score - self.threshold) / (abs(self.threshold) + 1e-6)), 2)

        return {
            "score": score,
            "threshold": self.threshold,
            "accepted": score >= self.threshold,
            "confidence": min(confidence, 1.0)
        }

    # BUG FIX: save() and load() were missing; called from engineering.py main()
    def save(self, path: str) -> None:
        """Сохраняет модель на диск."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "KeystrokeModel":
        """Загружает модель с диска."""
        with open(path, "rb") as f:
            return pickle.load(f)


# --- Вспомогательные функции для подготовки данных ---

def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Извлекает данные из результата transform_payload для обучения.
    """
    vectors = []
    feature_names = None

    for attempt in parsed_json.get("attempts", []):
        features_node = attempt.get("features", {})
        vector = features_node.get("feature_vector")
        names = features_node.get("feature_names")
        
        if vector and names:
            vectors.append(vector)
            if feature_names is None:
                feature_names = names

    return vectors, feature_names
