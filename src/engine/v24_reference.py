"""Faithful re-implementation of the v2.4 production blend for like-for-like comparison.

Uses the *unchanged* v2.4 feature builder, recency weights, Random Forest
hyper-parameters and the saved blend (70% RF / 30% season-to-date Poisson, factor 1.06).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEASON_SAMPLE_WEIGHTS
from src.historical_model import (FEATURE_COLUMNS as V24_FEATURES, _candidate_models, _fit, _predict_probabilities,
                                  build_historical_features, poisson_probabilities_from_features)

V24_ML_WEIGHT = 0.7
V24_HOME_FACTOR = 1.06


def v24_features(matches: pd.DataFrame) -> pd.DataFrame:
    return build_historical_features(matches)


def v24_predict(features: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series) -> dict[str, np.ndarray]:
    """Fit on ``train_mask`` rows and predict ``test_mask`` rows. Returns RF, Poisson and blend."""
    y = features["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy(dtype=int)
    X = features[V24_FEATURES].astype(float)
    weights = features["Season"].astype(str).map(SEASON_SAMPLE_WEIGHTS).fillna(0.5).to_numpy(dtype=float)
    model = _candidate_models()["Random Forest"]
    _fit(model, X[train_mask], y[train_mask], weights[train_mask])
    rf = _predict_probabilities(model, X[test_mask])
    poisson = poisson_probabilities_from_features(features.loc[test_mask], V24_HOME_FACTOR)
    blend = V24_ML_WEIGHT * rf + (1.0 - V24_ML_WEIGHT) * poisson
    return {"rf": rf, "poisson": poisson, "blend": blend / blend.sum(axis=1, keepdims=True)}
