"""
tests/test_pipeline.py
Unit tests for the Mall Customer Clustering pipeline.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import encode_gender, select_features, scale_features, apply_pca
from src.train import (
    compute_elbow, compute_silhouette, fit_kmeans,
    get_cluster_labels, cluster_summary,
)
from src.config import DATA_PATH, FEATURE_SETS, K_MIN, K_MAX, DEFAULT_K


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def enc_df(raw_df):
    return encode_gender(raw_df)


@pytest.fixture(scope="module")
def feat_names():
    return FEATURE_SETS["Income & Spending Score"]


@pytest.fixture(scope="module")
def X(enc_df, feat_names):
    return select_features(enc_df, feat_names)


@pytest.fixture(scope="module")
def scaled(X):
    return scale_features(X)


@pytest.fixture(scope="module")
def scaler(scaled):
    return scaled[0]


@pytest.fixture(scope="module")
def X_scaled(scaled):
    return scaled[1]


@pytest.fixture(scope="module")
def model(X_scaled):
    return fit_kmeans(X_scaled, DEFAULT_K)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_shape(self, raw_df):
        assert raw_df.shape == (200, 5)

    def test_no_missing(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_expected_columns(self, raw_df):
        for col in ["Customer_ID", "Gender", "Age", "Annual_Income", "Spending_Score"]:
            assert col in raw_df.columns


class TestPreprocessing:
    def test_gender_encoded_added(self, enc_df):
        assert "Gender_Encoded" in enc_df.columns

    def test_gender_encoded_binary(self, enc_df):
        assert set(enc_df["Gender_Encoded"].unique()).issubset({0, 1})

    def test_select_features_shape(self, X, feat_names):
        assert X.shape == (200, len(feat_names))

    def test_scaler_range(self, X_scaled):
        # StandardScaler: mean ~0, std ~1 (not bounded to [0,1])
        assert X_scaled.mean(axis=0).max() < 1e-9
        assert abs(X_scaled.std(axis=0).mean() - 1.0) < 0.05

    def test_pca_output_shape(self, X_scaled):
        _, X_pca, evr = apply_pca(X_scaled, n_components=2)
        assert X_pca.shape == (200, 2)
        assert len(evr) == 2
        assert 0 < evr.sum() <= 1.0


class TestTraining:
    def test_elbow_length(self, X_scaled):
        res = compute_elbow(X_scaled)
        assert len(res["k_values"]) == K_MAX - K_MIN + 1
        assert len(res["wcss"])     == K_MAX - K_MIN + 1

    def test_wcss_decreasing(self, X_scaled):
        res = compute_elbow(X_scaled)
        wcss = res["wcss"]
        assert all(wcss[i] >= wcss[i + 1] for i in range(len(wcss) - 1))

    def test_silhouette_range(self, X_scaled):
        res = compute_silhouette(X_scaled)
        for s in res["scores"]:
            assert -1.0 <= s <= 1.0

    def test_best_k_valid(self, X_scaled):
        res = compute_silhouette(X_scaled)
        assert K_MIN <= res["best_k"] <= K_MAX

    def test_kmeans_label_count(self, model, X_scaled):
        labels = get_cluster_labels(model, X_scaled)
        assert labels.shape[0] == 200
        assert set(labels) == set(range(DEFAULT_K))

    def test_cluster_summary_index(self, raw_df, model, X_scaled, feat_names):
        labels  = get_cluster_labels(model, X_scaled)
        summary = cluster_summary(raw_df, labels, feat_names)
        assert len(summary) == DEFAULT_K
        assert "Size" in summary.columns
