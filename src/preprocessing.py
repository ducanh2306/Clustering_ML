"""
Feature selection, encoding, and scaling for the clustering pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

from src.config import CUSTOMER_ID_COL, GENDER_COL, PCA_N_COMPONENTS
from src.logger import get_logger

logger = get_logger(__name__)


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    le = LabelEncoder()
    df["Gender_Encoded"] = le.fit_transform(df[GENDER_COL])
    logger.info("Gender encoded: %s", dict(zip(le.classes_, le.transform(le.classes_))))
    return df


def select_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in data: {missing}")
    X = df[feature_names].copy()
    logger.info("Selected features: %s  shape=%s", feature_names, X.shape)
    return X


def scale_features(X: pd.DataFrame) -> tuple:

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features scaled with StandardScaler.")
    return scaler, X_scaled


def apply_pca(X_scaled: np.ndarray, n_components: int = PCA_N_COMPONENTS) -> tuple:

    n_comp = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    logger.info(
        "PCA %d components — explained variance: %s  (total=%.3f)",
        n_comp, [f"{v:.3f}" for v in evr], evr.sum(),
    )
    return pca, X_pca, evr
