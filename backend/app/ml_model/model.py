"""
Keystroke Dynamics authentication model.

Pipeline: zero-variance drop → StandardScale → PCA → Mahalanobis one-class.

Why PCA before Mahalanobis?
  With ~462 raw features and ~30 enrollment samples the covariance matrix is
  severely under-determined (n << d).  PCA reduces dimensionality to
  k = min(n_samples - 2, pca_max_components) before fitting, ensuring n >> d
  and producing a stable, invertible covariance.

Why LedoitWolf instead of the manual shrinkage formula?
  sklearn's LedoitWolf computes the analytically optimal shrinkage coefficient
  (Ledoit & Wolf 2004) rather than the heuristic alpha = max(0, 1-(n-1)/d).
  It is well-validated and works better across varying n/d ratios.

Why a hybrid threshold?
  chi-squared gives the theoretically correct Mahalanobis radius for a
  multivariate normal.  But with small n the in-sample distances are
  downward-biased (we fit and evaluate on the same 30 points).  The hybrid
  threshold takes max(chi2_threshold, empirical_p95 * safety_factor) so
  genuine users are not rejected during initial verification.
"""

import pickle
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

from app.settings import env_settings


class KeystrokeModel:
    """
    PCA + Mahalanobis One-Class classifier for keystroke-dynamics authentication.

    Attributes
    ----------
    is_trained        : bool
    threshold         : float  – Mahalanobis radius below which an attempt is accepted.
    feature_names     : list   – Raw feature names surviving the zero-variance filter.
    """

    def __init__(self, confidence: float = 0.99, pca_max_components: int = 50):
        """
        Parameters
        ----------
        confidence : float
            Fraction of the chi-squared distribution used for the theoretical
            component of the acceptance threshold.  0.99 is appropriate after
            PCA reduction; the old 0.999 was set for 462-dim space and produced
            an unreasonably large threshold.
        pca_max_components : int
            Hard cap on PCA components.  The actual number used is
            min(n_samples - 2, pca_max_components, n_features_after_filter).
        """
        self.confidence = confidence
        self.pca_max_components = pca_max_components

        self.scaler: StandardScaler = StandardScaler()
        self.pca: Optional[PCA] = None

        self.is_trained: bool = False
        self.threshold: Optional[float] = None
        self.feature_names: Optional[List[str]] = None

        self._mu: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._keep_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, feature_vectors: List[List[float]], feature_names: List[str]) -> None:
        """
        Train on enrollment attempts.

        Steps
        -----
        1. Drop zero-variance features.
        2. StandardScale.
        3. PCA: reduce to k = min(n_samples - 2, pca_max_components, n_features).
        4. LedoitWolf regularised covariance in PCA space.
        5. Hybrid threshold: max(chi2_threshold, empirical_p95 * 2.5).
        """
        if not feature_vectors or len(feature_vectors) < 5:
            raise ValueError(
                f"Need at least 5 enrollment attempts, got {len(feature_vectors)}. "
                "Recommended: 30–40."
            )

        X = np.array(feature_vectors, dtype=float)

        # 1. Drop zero-variance features
        stds = X.std(axis=0)
        self._keep_mask = stds > 0
        X = X[:, self._keep_mask]
        self.feature_names = list(np.array(feature_names)[self._keep_mask])

        n_samples, n_features = X.shape

        # 2. StandardScale
        X = self.scaler.fit_transform(X)

        # 3. PCA — guarantees n_samples >> n_components for stable covariance
        n_components = min(n_samples - 2, self.pca_max_components, n_features)
        n_components = max(1, n_components)
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)

        # 4. LedoitWolf regularised covariance
        lw = LedoitWolf()
        lw.fit(X_pca)
        self._mu = X_pca.mean(axis=0)
        self._cov_inv = np.linalg.inv(lw.covariance_)

        # 5. Hybrid threshold
        #    a) Chi-squared: theoretically correct boundary for multivariate normal
        chi2_threshold = float(np.sqrt(stats.chi2.ppf(self.confidence, df=n_components)))

        #    b) Empirical: 95th-percentile of in-sample distances × safety factor.
        #       In-sample distances are biased low (fit and evaluate on same data),
        #       so the multiplier (2.5) adds the margin that LOO cross-validation
        #       would otherwise provide.
        train_dists = np.array([self._mahalanobis_raw(x) for x in X_pca])
        empirical_threshold = float(np.percentile(train_dists, 95)) * 2.5

        self.threshold = max(chi2_threshold, empirical_threshold)
        self.is_trained = True

    # ------------------------------------------------------------------
    def _mahalanobis_raw(self, x: np.ndarray) -> float:
        """Mahalanobis distance of a pre-projected, pre-scaled vector from mu."""
        delta = x - self._mu
        return float(np.sqrt(delta @ self._cov_inv @ delta))

    def _align_and_project(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Map feature dict → numpy array aligned with training features,
        apply StandardScaler, then project through PCA.
        """
        if self.feature_names is None or self.pca is None:
            raise RuntimeError("Model is not trained.")
        vec = np.array(
            [float(feature_dict.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        x_sc = self.scaler.transform(vec.reshape(1, -1))
        return self.pca.transform(x_sc)[0]

    # ------------------------------------------------------------------
    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Verify one attempt.

        Returns
        -------
        dict with keys:
            score      – Mahalanobis distance in PCA space (lower = more owner-like).
            threshold  – acceptance boundary.
            accepted   – bool.
            confidence – float in [0, 1], how comfortably within the boundary.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")

        x_pca = self._align_and_project(feature_dict)
        score = self._mahalanobis_raw(x_pca)

        conf = round(max(0.0, min(1.0, 1.0 - score / (self.threshold + 1e-9))), 3)
        accepted = (
            score <= self.threshold
            and conf * 100 >= env_settings.ThresholdOfConfidence
        )

        return {
            "score":      score,
            "threshold":  self.threshold,
            "accepted":   accepted,
            "confidence": conf,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "KeystrokeModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Helper for training data extraction
# ---------------------------------------------------------------------------

def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Extract (vectors, feature_names) from the output of transform_payload().

    Different attempts may produce different n-gram features (digraphs/trigraphs
    depend on the exact keys pressed, including typos and corrections).  Collecting
    feature_vector from the first attempt and reusing its feature_names for all
    others results in vectors of different lengths → inhomogeneous numpy array.

    Fix: collect flat_features dicts from every attempt, compute the *union* of all
    feature names, then rebuild each vector by looking up names in the dict (missing
    features → 0.0).  The resulting matrix is always rectangular.
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
