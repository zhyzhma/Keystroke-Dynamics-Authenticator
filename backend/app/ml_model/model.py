"""
Keystroke Dynamics authentication model.

Algorithm: Scaled Manhattan Distance to enrollment centroid.

This is the standard reference algorithm for fixed-text keystroke dynamics,
described and benchmarked in:
  Killourhy & Maxion (2009) "Comparing Anomaly-Detection Algorithms for
  Keystroke Dynamics", IEEE DSN.  It consistently ranks among the top
  performers on fixed-text datasets.

Given an enrollment matrix X (n_attempts × d_features) and a verification
vector x, the score is:

    score = (1/d) * Σ_i  |x_i − μ_i| / max(σ_i, ε)

where μ_i and σ_i are the per-feature enrollment mean and sample standard
deviation, and ε is a small floor that prevents division-by-zero when a
feature is perfectly consistent across all enrollment attempts.

Lower score means the attempt is more similar to the enrolled profile.
Threshold is set via Leave-One-Out (LOO) calibration:

    threshold = mean(loo_scores) + k * std(loo_scores)

where k defaults to 2.5 and loo_scores[i] is the score of attempt i
evaluated against the model trained on all OTHER attempts.  LOO scores
are unbiased estimates of out-of-sample owner performance, so the
threshold correctly accounts for genuine variation across sessions.

The sigma floor is set to max(p25(sigma), eps) to prevent near-zero-
variance features from dominating the Scaled Manhattan sum.

The ThresholdOfConfidence setting (0–100) acts as an additional strictness
knob: it requires confidence = (1 − score/threshold) × 100 ≥ the setting,
effectively tightening the boundary without re-training the model.
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Optional

from app.settings import env_settings


class KeystrokeModel:
    """
    Fixed-text keystroke dynamics authenticator.

    Attributes
    ----------
    is_trained    : bool
    threshold     : float  – Scaled Manhattan acceptance boundary.
    feature_names : list   – Feature names from enrollment (phrase-aligned).
    phrase        : str    – Phrase used during enrollment (informational).
    _mu           : ndarray – Per-feature enrollment mean.
    _sigma        : ndarray – Per-feature enrollment std (floored at _EPS).
    _enrollment   : ndarray – Raw enrollment matrix (kept for diagnostics).
    """

    _EPS = 1e-6          # absolute sigma floor to avoid division by zero
    _SIGMA_K = 2.5       # threshold = mean_loo + _SIGMA_K * std_loo

    def __init__(self):
        self.is_trained:    bool                 = False
        self.threshold:     Optional[float]      = None
        self.feature_names: Optional[List[str]]  = None
        self.phrase:        Optional[str]        = None

        self._mu:         Optional[np.ndarray] = None
        self._sigma:      Optional[np.ndarray] = None
        self._enrollment: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        feature_vectors: List[List[float]],
        feature_names:   List[str],
        phrase:          str = "",
    ) -> None:
        """
        Train on enrollment attempts.

        All vectors must have the same length (guaranteed by extract_training_data
        when the phrase does not change between attempts).

        Steps
        -----
        1. Build enrollment matrix X (n × d).
        2. Compute per-feature mean μ and sample std σ (floored at _EPS).
        3. Compute Scaled Manhattan distance for every training vector.
        4. Set threshold = mean(scores) + _SIGMA_K × std(scores).
        """
        if not feature_vectors or len(feature_vectors) < 5:
            raise ValueError(
                f"Need at least 5 valid enrollment attempts, got {len(feature_vectors)}. "
                "Recommended: 30–40."
            )

        X = np.array(feature_vectors, dtype=float)   # (n, d)
        self.feature_names = list(feature_names)
        self.phrase        = phrase
        self._enrollment   = X

        # Per-feature statistics
        self._mu    = X.mean(axis=0)
        sigma_raw   = X.std(axis=0, ddof=1)           # sample std
        # p25 floor: prevents near-zero-variance features from dominating the sum
        positive_sigmas = sigma_raw[sigma_raw > 0]
        sigma_floor = float(np.percentile(positive_sigmas, 25)) if len(positive_sigmas) > 0 else self._EPS
        self._sigma = np.maximum(sigma_raw, max(sigma_floor, self._EPS))

        # LOO threshold: unbiased estimate of out-of-sample owner performance
        loo_scores = self._compute_loo_scores(X)
        t_mean = float(loo_scores.mean())
        t_std  = float(loo_scores.std(ddof=1)) if len(loo_scores) > 1 else 0.0
        self.threshold = t_mean + self._SIGMA_K * t_std

        self.is_trained = True

    # ------------------------------------------------------------------
    def _compute_loo_scores(self, X: np.ndarray) -> np.ndarray:
        """Leave-one-out Scaled Manhattan scores for each enrollment attempt."""
        n = len(X)
        scores = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_tr = X[mask]
            mu_i = X_tr.mean(axis=0)
            raw_sig = X_tr.std(axis=0, ddof=1)
            pos = raw_sig[raw_sig > 0]
            floor_i = float(np.percentile(pos, 25)) if len(pos) > 0 else self._EPS
            sig_i = np.maximum(raw_sig, max(floor_i, self._EPS))
            scores[i] = float(np.mean(np.abs(X[i] - mu_i) / sig_i))
        return scores

    def _score(self, x: np.ndarray) -> float:
        """Scaled Manhattan distance of a single row vector from the enrollment mean."""
        return float(np.mean(np.abs(x - self._mu) / self._sigma))

    def _batch_score(self, X: np.ndarray) -> np.ndarray:
        """Vectorised Scaled Manhattan for a (n × d) matrix."""
        return np.mean(np.abs(X - self._mu) / self._sigma, axis=1)

    def _align_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Map a feature dict → 1-D numpy array aligned to self.feature_names.
        Features absent in the dict default to 0.0.
        """
        if self.feature_names is None:
            raise RuntimeError("Model has no feature names stored.")
        return np.array(
            [float(feature_dict.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )

    # ------------------------------------------------------------------
    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Verify one attempt.

        Returns
        -------
        dict
            score      – Scaled Manhattan distance (lower = more owner-like).
            threshold  – Acceptance boundary (set during fit).
            accepted   – True if score ≤ threshold AND confidence ≥ ThresholdOfConfidence.
            confidence – float ∈ [0, 1]: how comfortably within the boundary.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")

        x     = self._align_vector(feature_dict)
        score = self._score(x)

        # confidence: 1.0 at score=0, 0.0 at score=threshold, negative beyond
        conf = round(max(0.0, min(1.0, 1.0 - score / (self.threshold + self._EPS))), 3)

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
# Helper: extract training data from transform_payload output
# ---------------------------------------------------------------------------

def extract_training_data(parsed_json: Dict[str, Any]):
    """
    Extract (vectors, feature_names) from the output of transform_payload().

    Only includes attempts where "valid" is True (typed text matched phrase).
    With phrase-aligned features all valid attempts have the same feature names,
    so no union/padding is needed.

    Returns
    -------
    (vectors, feature_names)  or  ([], []) when no valid attempts exist.
    """
    flat_list: List[Dict[str, float]] = []

    for attempt in parsed_json.get("attempts", []):
        feats = attempt.get("features", {})
        if feats.get("valid") and feats.get("flat_features"):
            flat_list.append(feats["flat_features"])

    if not flat_list:
        return [], []

    # All valid attempts share the same feature names (phrase-aligned)
    all_names = sorted(flat_list[0].keys())
    vectors   = [[f.get(name, 0.0) for name in all_names] for f in flat_list]

    return vectors, all_names
