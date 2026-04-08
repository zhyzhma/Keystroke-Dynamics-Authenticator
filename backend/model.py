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

    # =========================
    # TRAINING
    # =========================
    def fit(self, feature_vectors: List[List[float]], feature_names: List[str]):
        """
        feature_vectors: List of vectors from attempts
        feature_names: consistent ordering of features
        """
        X = np.array(feature_vectors)

        # scaling
        X_scaled = self.scaler.fit_transform(X)

        # train model
        self.model.fit(X_scaled)

        # save feature order
        self.feature_names = feature_names

        # compute scores on training data
        scores = self.model.decision_function(X_scaled)

        # threshold = минимальный score (можно улучшить позже)
        self.threshold = float(np.min(scores))

        self.is_trained = True

    # =========================
    # SCORING
    # =========================
    def score(self, feature_vector: List[float]) -> float:
        if not self.is_trained:
            raise RuntimeError("Model is not trained")

        x = np.array(feature_vector).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        return float(self.model.decision_function(x_scaled)[0])

    # =========================
    # PREDICTION
    # =========================
    def predict(self, feature_vector: List[float]) -> Dict[str, Any]:
        score = self.score(feature_vector)

        return {
            "score": score,
            "threshold": self.threshold,
            "accepted": score >= self.threshold
        }

    # =========================
    # SAVE / LOAD
    # =========================
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
    Берёт JSON после feature extraction
    Возвращает:
      - список feature_vectors
      - feature_names
    """
    vectors = []
    feature_names = None

    for attempt in parsed_json["attempts"]:
        features = attempt["features"]

        vectors.append(features["feature_vector"])

        if feature_names is None:
            feature_names = features["feature_names"]

    return vectors, feature_names


def extract_single_vector(parsed_json: Dict[str, Any]) -> List[float]:
    """
    Берёт первый attempt (например для login)
    """
    return parsed_json["attempts"][0]["features"]["feature_vector"]