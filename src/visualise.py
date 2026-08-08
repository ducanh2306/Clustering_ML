"""
All reusable chart-generation functions for the Streamlit app.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import KMeans

from src.logger import get_logger

logger = get_logger(__name__)

# Consistent colour palette
PALETTE = "tab10"


def plot_elbow(k_values: list, wcss: list) -> plt.Figure:
    """Within-cluster sum of squares vs k (elbow curve)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, wcss, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Method — Optimal k")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_silhouette_scores(k_values: list, scores: list, best_k: int) -> plt.Figure:
    """Mean silhouette coefficient vs k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if k == best_k else "#3498db" for k in k_values]
    ax.bar(k_values, scores, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis — Optimal k")
    ax.set_xticks(k_values)
    ax.axhline(max(scores), color="grey", linestyle="--", linewidth=0.8)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label=f"Best k={best_k}"),
        ],
        loc="upper right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_clusters_2d(
    X_orig: pd.DataFrame,
    labels: np.ndarray,
    model: KMeans,
    scaler,
    feature_names: list,
) -> plt.Figure:
    
    if len(feature_names) != 2:
        raise ValueError("plot_clusters_2d requires exactly 2 features.")

    feat_x, feat_y = feature_names
    n_clusters = model.n_clusters
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 0.9, n_clusters))

    # Inverse-transform centroids back to original scale
    centroids_orig = scaler.inverse_transform(model.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            X_orig.loc[mask, feat_x],
            X_orig.loc[mask, feat_y],
            c=[colors[k]], label=f"Cluster {k}", s=60, alpha=0.75, edgecolors="white",
        )
    ax.scatter(
        centroids_orig[:, 0], centroids_orig[:, 1],
        marker="X", s=220, c="black", zorder=5, label="Centroids",
    )
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"KMeans Clusters (k={n_clusters})  —  {feat_x} vs {feat_y}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_clusters_pca(X_pca: np.ndarray, labels: np.ndarray,
                      evr: np.ndarray, n_clusters: int) -> plt.Figure:
    """2-D PCA projection coloured by cluster label."""
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 0.9, n_clusters))

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(n_clusters):
        mask = labels == k
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors[k]], label=f"Cluster {k}", s=60, alpha=0.75, edgecolors="white",
        )
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} variance)" if len(evr) > 1 else "PC2")
    ax.set_title(f"PCA Projection — {n_clusters} Clusters")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_silhouette_diagram(X_scaled: np.ndarray, labels: np.ndarray,
                             sample_scores: np.ndarray, n_clusters: int) -> plt.Figure:
 
    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10
    colors = plt.get_cmap(PALETTE)(np.linspace(0, 0.9, n_clusters))

    for k in range(n_clusters):
        cluster_scores = np.sort(sample_scores[labels == k])
        size = cluster_scores.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_scores,
            facecolor=colors[k], edgecolor=colors[k], alpha=0.8,
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(k), fontsize=8, color="black")
        y_lower = y_upper + 10

    mean_score = sample_scores.mean()
    ax.axvline(mean_score, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot (k={n_clusters}, mean={mean_score:.3f})")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_feature_distributions(df: pd.DataFrame, labels: np.ndarray,
                                feature_names: list) -> plt.Figure:
    """Box-plot of each feature broken down by cluster."""
    df_plot = df[feature_names].copy()
    df_plot["Cluster"] = labels.astype(str)

    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, feature_names):
        sns.boxplot(
            data=df_plot, x="Cluster", y=feat,
            palette=PALETTE, ax=ax, width=0.5,
        )
        ax.set_title(feat)
        ax.set_xlabel("Cluster")
    fig.suptitle("Feature Distribution by Cluster", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_gender_distribution(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    """Stacked bar of Male / Female proportion per cluster."""
    df_plot = df[["Gender"]].copy()
    df_plot["Cluster"] = labels

    counts = (
        df_plot.groupby(["Cluster", "Gender"])
        .size()
        .unstack(fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    proportions.plot(kind="bar", stacked=True, ax=ax,
                     color=["#e91e8c", "#2196F3"], edgecolor="white")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_title("Gender Distribution per Cluster")
    ax.legend(title="Gender", loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame, labels: np.ndarray) -> plt.Figure:
    """Seaborn pairplot of numeric features coloured by cluster."""
    num_cols = ["Age", "Annual_Income", "Spending_Score"]
    df_plot = df[num_cols].copy()
    df_plot["Cluster"] = labels.astype(str)

    g = sns.pairplot(
        df_plot, hue="Cluster", palette=PALETTE,
        diag_kind="kde", plot_kws={"alpha": 0.6, "s": 30},
    )
    g.fig.suptitle("Pairplot — All Numeric Features", y=1.02)
    return g.fig
