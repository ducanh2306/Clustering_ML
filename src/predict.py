"""
Assign a new customer to an existing KMeans cluster.
"""

import numpy as np
import pandas as pd

from src.train import load_model
from src.logger import get_logger

logger = get_logger(__name__)


def predict_cluster(raw_input: dict, feature_names: list, scaler) -> dict:

    try:
        model = load_model()
    except FileNotFoundError:
        logger.error("No saved model found. Train a model first.")
        raise

    df_input = pd.DataFrame([{f: raw_input[f] for f in feature_names}])
    X_scaled = scaler.transform(df_input)

    cluster   = int(model.predict(X_scaled)[0])
    distances = np.linalg.norm(model.cluster_centers_ - X_scaled, axis=1).tolist()

    logger.info(
        "New customer → Cluster %d  (dist to centroid=%.4f)",
        cluster, distances[cluster],
    )
    return {"cluster": cluster, "distances": distances}
