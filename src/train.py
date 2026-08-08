"""
KMeans clustering: elbow method, silhouette analysis, model fitting, persistence.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from src.config import (
    K_MIN, K_MAX, DEFAULT_K, RANDOM_STATE,
    KMEANS_INIT, KMEANS_N_INIT, MODEL_PATH,
)
from src.logger import get_logger

logger = get_logger(__name__)



def compute_elbow(X_scaled: np.ndarray) -> dict:
    
    k_values, wcss = [], []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(
            n_clusters=k,
            init=KMEANS_INIT,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        )
        km.fit(X_scaled)
        k_values.append(k)
        wcss.append(km.inertia_)
        logger.debug("Elbow  k=%d  inertia=%.2f", k, km.inertia_)

    logger.info("Elbow sweep complete for k=%d…%d.", K_MIN, K_MAX)
    return {"k_values": k_values, "wcss": wcss}


def compute_silhouette(X_scaled: np.ndarray) -> dict:

    k_values, scores = [], []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(
            n_clusters=k,
            init=KMEANS_INIT,
            n_init=KMEANS_N_INIT,
            random_state=RANDOM_STATE,
        )
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        k_values.append(k)
        scores.append(score)
        logger.debug("Silhouette  k=%d  score=%.4f", k, score)

    best_k = k_values[int(np.argmax(scores))]
    logger.info(
        "Silhouette sweep complete. Best k=%d (score=%.4f).",
        best_k, max(scores),
    )
    return {"k_values": k_values, "scores": scores, "best_k": best_k}


#Model fitting 

def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:

    km = KMeans(
        n_clusters=k,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )
    km.fit(X_scaled)
    logger.info(
        "KMeans fitted: k=%d  inertia=%.4f  iterations=%d",
        k, km.inertia_, km.n_iter_,
    )
    return km


def get_cluster_labels(model: KMeans, X_scaled: np.ndarray) -> np.ndarray:
    """Return integer cluster label for every sample."""
    return model.predict(X_scaled)


def get_silhouette_per_sample(X_scaled: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample silhouette coefficients (for the silhouette plot)."""
    return silhouette_samples(X_scaled, labels)


# Persistence 

def save_model(model: KMeans, path: str = MODEL_PATH) -> None:
    """Pickle the fitted KMeans model."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)


def load_model(path: str = MODEL_PATH) -> KMeans:
    """Load a pickled KMeans model."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded ← %s", path)
    return model


#Summary stats

def cluster_summary(df: pd.DataFrame, labels: np.ndarray,
                    feature_names: list) -> pd.DataFrame:
    df_copy = df[feature_names].copy()
    df_copy["Cluster"] = labels
    summary = df_copy.groupby("Cluster").agg(
        Size=("Cluster", "count"),
        **{f: (f, "mean") for f in feature_names},
    ).round(2)
    logger.info("Cluster summary:\n%s", summary.to_string())
    return summary
