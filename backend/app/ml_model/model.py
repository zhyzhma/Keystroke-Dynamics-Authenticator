"""
Keystroke Dynamics authentication model.

Uses a Mahalanobis-distance One-Class classifier instead of RBF One-Class SVM.

Why not One-Class SVM with RBF kernel?
  After StandardScaler the feature matrix has ~462 dimensions and ~30 samples.
  With gamma='scale' = 1/462 the RBF kernel value between ANY two scaled vectors
  is exp(-gamma * ||xi-xj||^2) ≈ exp(-0.002 * 34^2) ≈ 0.10 -- near-zero for
  ALL pairs, so the SVM cannot distinguish the owner's cluster from background noise.
  No nu or gamma tuning fixes this: it is a manifestation of the curse of
  dimensionality in the RBF RKHS with n << d.

Mahalanobis distance naturally handles correlated high-dimensional data:
  - Fit: compute mean mu and regularised covariance Sigma from enrollment vectors.
  - Score: d_M(x) = sqrt((x-mu)^T Sigma^-1 (x-mu)).
  - Threshold: chi-squared critical value at chosen confidence level (default 99%).
  - Owner scores cluster near 0; impostors with different timing are far away.

The public interface (fit / predict / save / load) is identical to what the rest
of the codebase expects, so security.py and engineering.py are unchanged.
"""

import pickle
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from app.settings import env_settings

class KeystrokeModel:
    """
    Mahalanobis One-Class classifier for keystroke-dynamics authentication.

    Attributes
    ----------
    is_trained : bool
    threshold  : float   – Mahalanobis distance below which an attempt is accepted.
    feature_names : list – Names of the (filtered) features used by the model.
    """

    def __init__(self, confidence: float = 0.999):
        """
        Parameters
        ----------
        confidence : float
            Fraction of the chi-squared distribution used to set the acceptance
            threshold.  0.999 means we accept the top-99.9% of the chi-squared
            distribution, i.e. attempts within the 99.9% confidence ellipsoid
            of the owner's timing distribution.
        """
        self.confidence   = confidence
        self.scaler       = StandardScaler()

        self.is_trained:    bool                = False
        self.threshold:     Optional[float]     = None
        self.feature_names: Optional[List[str]] = None

        # Fitted distribution parameters (set by fit())
        self._mu:           Optional[np.ndarray] = None
        self._cov_inv:      Optional[np.ndarray] = None
        self._keep_mask:    Optional[np.ndarray] = None  # non-zero-variance column mask

    # ------------------------------------------------------------------
    def fit(self, feature_vectors: List[List[float]], feature_names: List[str]) -> None:
        """
        Train on enrollment attempts.

        Steps
        -----
        1. Drop zero-variance columns (constant features add noise to cov^-1).
        2. StandardScale the remaining columns.
        3. Compute mean and regularised covariance of the scaled matrix.
        4. Set threshold = sqrt(chi2.ppf(confidence, df=n_features)).
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
        X_filtered = X[:, self._keep_mask]
        all_names  = np.array(feature_names)
        self.feature_names = list(all_names[self._keep_mask])

        n_samples, n_features = X_filtered.shape

        # 2. Scale
        X_scaled = self.scaler.fit_transform(X_filtered)

        # 3. Mean + regularised covariance
        self._mu = X_scaled.mean(axis=0)

        # Ledoit-Wolf shrinkage regularisation: stable inverse even when n < d
        cov_raw = np.cov(X_scaled, rowvar=False)          # (d, d)
        # Shrink toward identity: Sigma = (1-alpha)*cov + alpha*I
        alpha   = max(0.0, 1.0 - (n_samples - 1) / n_features)
        cov_reg = (1.0 - alpha) * cov_raw + alpha * np.eye(n_features)
        self._cov_inv = np.linalg.inv(cov_reg)

        # 4. Threshold from chi-squared distribution
        # sqrt(chi2.ppf(p, df)) gives the Mahalanobis radius enclosing
        # fraction p of a multivariate normal distribution.
        chi2_val       = stats.chi2.ppf(self.confidence, df=n_features)
        self.threshold = float(np.sqrt(chi2_val))

        self.is_trained = True

    # ------------------------------------------------------------------
    def _mahalanobis(self, x: np.ndarray) -> float:
        """Return the Mahalanobis distance of a single scaled row vector from mu."""
        delta = x - self._mu
        return float(np.sqrt(delta @ self._cov_inv @ delta))

    def _align_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Map feature dict → 1-D array aligned with self.feature_names."""
        if self.feature_names is None:
            raise RuntimeError("Model has no feature names stored.")
        vec = np.array(
            [float(feature_dict.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        return vec

    # ------------------------------------------------------------------
    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Verify one attempt.

        Returns
        -------
        dict with keys:
            score      – raw Mahalanobis distance (lower = more similar to owner).
            threshold  – acceptance boundary.
            accepted   – bool, True if score <= threshold.
            confidence – float in [0, 1], how comfortably inside the boundary.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")

        vec     = self._align_vector(feature_dict)
        x_sc    = self.scaler.transform(vec.reshape(1, -1))[0]
        score   = self._mahalanobis(x_sc)
        accepted = score <= self.threshold and (1.0 - score / (self.threshold + 1e-9))*100 >= env_settings.ThresholdOfConfidence

        # confidence: 1.0 at score=0, 0.0 at score=threshold, negative beyond
        conf = round(max(0.0, min(1.0, 1.0 - score / (self.threshold + 1e-9))), 3)

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

    Different attempts may produce different n-gram features (digraphs / trigraphs
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
