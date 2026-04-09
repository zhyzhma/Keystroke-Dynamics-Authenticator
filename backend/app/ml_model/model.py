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
        # Обучаем скалер
        X_scaled = self.scaler.fit_transform(X)
        # Обучаем модель
        self.model.fit(X_scaled)
        
        self.feature_names = feature_names

        # Рассчитываем порог: 5-й перцентиль защищает от случайных опечаток при обучении
        scores = self.model.decision_function(X_scaled)
        self.threshold = float(np.percentile(scores, 5)) 
        
        self.is_trained = True

    def _align_vector(self, feature_dict: Dict[str, float]) -> List[float]:
        """
        Критически важный метод: превращает словарь признаков от фронтенда
        в упорядоченный список, который понимает модель.
        """
        if not self.feature_names:
            raise RuntimeError("Model has no feature names stored.")
        
        # Если при верификации какого-то триграфа или диграфа нет — ставим 0.0
        return [float(feature_dict.get(name, 0.0)) for name in self.feature_names]

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Верификация по словарю признаков."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
        
        # Выравниваем признаки по эталону, который был при обучении
        vector = self._align_vector(feature_dict)
        x = np.array(vector).reshape(1, -1)
        
        # Важно: используем тот же скалер, что и при обучении
        x_scaled = self.scaler.transform(x)
        score = float(self.model.decision_function(x_scaled)[0])

        # Расчет уверенности (confidence)
        # Чем выше score над порогом, тем выше уверенность
        confidence = round(max(0, (score - self.threshold) / (abs(self.threshold) + 1e-6)), 2)

        return {
            "score": score,
            "threshold": self.threshold,
            "accepted": score >= self.threshold,
            "confidence": min(confidence, 1.0) # Ограничиваем сверху единицей
        }

# --- Вспомогательные функции для подготовки данных ---

def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Извлекает данные из результата transform_payload для обучения.
    """
    vectors = []
    feature_names = None

    for attempt in parsed_json.get("attempts", []):
        features_node = attempt.get("features", {})
        
        # Используем именно feature_vector, так как в engineering.py 
        # он уже собран в правильном порядке вместе с именами
        vector = features_node.get("feature_vector")
        names = features_node.get("feature_names")
        
        if vector and names:
            vectors.append(vector)
            if feature_names is None:
                feature_names = names

    return vectors, feature_names