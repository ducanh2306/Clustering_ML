"""
app.py
Streamlit web application — Mall Customer Segmentation (K-Means Clustering).
Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    DATA_PATH, MODEL_PATH, DEFAULT_K, K_MIN, K_MAX, FEATURE_SETS,
)
from src.logger import get_logger
from src.data_loader import load_data
from src.preprocessing import encode_gender, select_features, scale_features, apply_pca
from src.train import (
    compute_elbow, compute_silhouette,
    fit_kmeans, get_cluster_labels, get_silhouette_per_sample,
    cluster_summary, save_model, load_model,
)
from src.predict import predict_cluster
from src.visualise import (
    plot_elbow, plot_silhouette_scores, plot_clusters_2d,
    plot_clusters_pca, plot_silhouette_diagram,
    plot_feature_distributions, plot_gender_distribution, plot_pairplot,
)

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def get_raw_data():
    return load_data(DATA_PATH)


@st.cache_data
def run_elbow_silhouette(feature_key: str):
    """Run elbow + silhouette sweep (cached per feature set)."""
    df   = load_data(DATA_PATH)
    df   = encode_gender(df)
    feat = FEATURE_SETS[feature_key]
    X    = select_features(df, feat)
    _, X_scaled = scale_features(X)
    elbow_res = compute_elbow(X_scaled)
    sil_res   = compute_silhouette(X_scaled)
    return elbow_res, sil_res


@st.cache_resource
def run_clustering(feature_key: str, k: int):
    """Fit KMeans and return everything needed for the app (cached)."""
    df      = load_data(DATA_PATH)
    df      = encode_gender(df)
    feat    = FEATURE_SETS[feature_key]
    X       = select_features(df, feat)
    scaler, X_scaled = scale_features(X)

    # PCA only if more than 2 features
    pca, X_pca, evr = apply_pca(X_scaled)

    model  = fit_kmeans(X_scaled, k)
    labels = get_cluster_labels(model, X_scaled)
    sil_samples = get_silhouette_per_sample(X_scaled, labels)
    summary = cluster_summary(df, labels, feat)

    save_model(model)

    return {
        "df": df,
        "X": X,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "evr": evr,
        "scaler": scaler,
        "model": model,
        "labels": labels,
        "sil_samples": sil_samples,
        "summary": summary,
        "feat": feat,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🛍️ Customer Segmentation")
page = st.sidebar.radio(
    "Navigate",
    ["Data Explorer", "Optimal K", "Clustering", "Predict"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")

feature_key = st.sidebar.selectbox("Feature Set", list(FEATURE_SETS.keys()))
k_value     = st.sidebar.slider("Number of Clusters (k)", K_MIN, K_MAX, DEFAULT_K)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Explorer
# ════════════════════════════════════════════════════════════════════════════
if page == "Data Explorer":
    st.title("📊 Mall Customer Data Explorer")
    st.markdown(
        "Explore the **Mall Customers** dataset — 200 customers described by "
        "age, annual income, and spending score."
    )

    df = get_raw_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers",     df.shape[0])
    c2.metric("Features",      df.shape[1] - 1)
    c3.metric("Avg Income",    f"${df['Annual_Income'].mean():.0f}k")
    c4.metric("Avg Spending",  f"{df['Spending_Score'].mean():.0f} / 100")

    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.subheader("Gender Distribution")
    import matplotlib.pyplot as plt
    vc = df["Gender"].value_counts()
    fig_g, ax_g = plt.subplots(figsize=(4, 3))
    ax_g.pie(
        vc.values, labels=vc.index,
        colors=["#2196F3", "#e91e8c"],
        autopct="%1.1f%%", startangle=90,
    )
    ax_g.set_title("Gender Split")
    st.pyplot(fig_g)

    st.subheader("Feature Distributions")
    num_cols = ["Age", "Annual_Income", "Spending_Score"]
    fig_d, axes = plt.subplots(1, 3, figsize=(13, 3))
    for ax, col in zip(axes, num_cols):
        ax.hist(df[col], bins=20, color="#9b59b6", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("")
    fig_d.tight_layout()
    st.pyplot(fig_d)

    st.subheader("Correlation Heatmap")
    import seaborn as sns
    fig_c, ax_c = plt.subplots(figsize=(5, 4))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax_c, linewidths=0.5, vmin=-1, vmax=1)
    ax_c.set_title("Pearson Correlation")
    fig_c.tight_layout()
    st.pyplot(fig_c)

    st.subheader("Pairplot")
    df_enc = encode_gender(df)
    fig_pp = plot_pairplot(df_enc, np.zeros(len(df_enc), dtype=int))
    # Suppress the dummy cluster colouring — just show structure
    st.pyplot(fig_pp)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Optimal K
# ════════════════════════════════════════════════════════════════════════════
elif page == "Optimal K":
    st.title("🔍 Finding the Optimal Number of Clusters")
    st.markdown(
        f"Sweeping k = {K_MIN} … {K_MAX} using the **Elbow Method** and "
        f"**Silhouette Analysis** on feature set: **{feature_key}**."
    )

    with st.spinner("Computing elbow and silhouette curves …"):
        elbow_res, sil_res = run_elbow_silhouette(feature_key)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Curve (WCSS)")
        st.pyplot(plot_elbow(elbow_res["k_values"], elbow_res["wcss"]))
        st.caption(
            "Look for the 'elbow' — the point where adding more clusters gives "
            "diminishing returns in WCSS reduction."
        )

    with col2:
        st.subheader("Silhouette Scores")
        st.pyplot(
            plot_silhouette_scores(
                sil_res["k_values"], sil_res["scores"], sil_res["best_k"]
            )
        )
        st.caption(
            f"Higher silhouette → better-separated clusters. "
            f"Best k = **{sil_res['best_k']}** (score = {max(sil_res['scores']):.4f})."
        )

    st.subheader("Scores Table")
    tbl = pd.DataFrame({
        "k":               elbow_res["k_values"],
        "WCSS (Inertia)":  [f"{w:.2f}" for w in elbow_res["wcss"]],
        "Silhouette":      [f"{s:.4f}" for s in sil_res["scores"]],
    })
    st.dataframe(tbl.set_index("k"), use_container_width=True)

    st.info(
        f"💡 **Recommendation:** Use k = **{sil_res['best_k']}** (highest silhouette) "
        f"or inspect the elbow curve for your business context. "
        f"Adjust k with the sidebar slider.",
        icon="💡",
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Clustering
# ════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.title(f"🤖 K-Means Clustering  (k = {k_value})")
    st.markdown(
        f"Feature set: **{feature_key}** | "
        f"Features: `{'`, `'.join(FEATURE_SETS[feature_key])}`"
    )

    with st.spinner(f"Fitting KMeans with k={k_value} …"):
        res = run_clustering(feature_key, k_value)

    df       = res["df"]
    labels   = res["labels"]
    feat     = res["feat"]
    model    = res["model"]
    scaler   = res["scaler"]
    X_scaled = res["X_scaled"]
    X        = res["X"]

    # ── Metrics ──────────────────────────────────────────────────────────
    from sklearn.metrics import silhouette_score as _sil
    overall_sil = _sil(X_scaled, labels)

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters",         k_value)
    c2.metric("Inertia (WCSS)",   f"{model.inertia_:.2f}")
    c3.metric("Silhouette Score", f"{overall_sil:.4f}")

    # ── Cluster summary table ────────────────────────────────────────────
    st.subheader("Cluster Summary")
    st.dataframe(res["summary"], use_container_width=True)

    # ── Labelled dataset download ────────────────────────────────────────
    df_out = df.copy()
    df_out["Cluster"] = labels
    st.subheader("Labelled Dataset")
    st.dataframe(df_out.head(20), use_container_width=True)

    csv_bytes = df_out.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download labelled CSV",
        data=csv_bytes,
        file_name="mall_customers_clustered.csv",
        mime="text/csv",
    )

    # ── Visualisations ───────────────────────────────────────────────────
    st.subheader("Cluster Visualisations")

    tab_labels = ["2-D Scatter", "PCA Projection", "Silhouette Diagram",
                  "Feature Distributions", "Gender Distribution"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if len(feat) == 2:
            st.pyplot(plot_clusters_2d(X, labels, model, scaler, feat))
        else:
            st.info(
                "2-D scatter is available only for 2-feature sets. "
                "Use the PCA Projection tab instead."
            )

    with tabs[1]:
        if res["X_pca"].shape[1] >= 2:
            st.pyplot(
                plot_clusters_pca(res["X_pca"], labels, res["evr"], k_value)
            )
            st.caption(
                f"PCA explains {res['evr'][:2].sum():.1%} of total variance."
            )
        else:
            st.info("PCA projection requires ≥ 2 features.")

    with tabs[2]:
        st.pyplot(
            plot_silhouette_diagram(X_scaled, labels, res["sil_samples"], k_value)
        )
        st.caption(
            "Bars to the right of the red dashed line (mean silhouette) "
            "indicate well-separated samples."
        )

    with tabs[3]:
        st.pyplot(plot_feature_distributions(df, labels, feat))

    with tabs[4]:
        st.pyplot(plot_gender_distribution(df, labels))

    # ── Per-cluster stats expander ────────────────────────────────────────
    with st.expander("📋 Per-cluster statistics"):
        for k in range(k_value):
            mask = labels == k
            st.markdown(f"**Cluster {k}** — {mask.sum()} customers")
            extra_cols = [c for c in ["Gender", "Age"] if c not in feat]
            display_cols = feat + extra_cols
            st.dataframe(
                df[mask][display_cols].describe().round(2),
                use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Predict
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Assign a New Customer to a Cluster")
    st.markdown(
        "Enter a new customer's details. The app will assign them to the "
        "nearest cluster using the last-trained model."
    )

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "No trained model found. Go to **🤖 Clustering** to train one first.",
            icon="⚠️",
        )
        st.stop()

    # We need the scaler — re-run clustering silently (cached)
    with st.spinner("Loading model …"):
        res = run_clustering(feature_key, k_value)

    scaler = res["scaler"]
    feat   = res["feat"]
    labels = res["labels"]
    df     = res["df"]

    with st.form("predict_form"):
        st.subheader("Customer Details")
        col1, col2 = st.columns(2)

        with col1:
            age            = st.slider("Age",            18, 70, 35)
            annual_income  = st.slider("Annual Income (k$)", 15, 140, 60)

        with col2:
            spending_score = st.slider("Spending Score (1–100)", 1, 100, 50)
            gender         = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("Click here to predict Cluster", type="primary")

    if submitted:
        raw_input = {
            "Age":            age,
            "Annual_Income":  annual_income,
            "Spending_Score": spending_score,
            "Gender_Encoded": 1 if gender == "Male" else 0,
        }

        # Only pass features the model was trained on
        model_input = {f: raw_input[f] for f in feat if f in raw_input}

        try:
            result  = predict_cluster(model_input, feat, scaler)
            cluster = result["cluster"]

            import matplotlib.pyplot as plt

            st.success(f"This customer belongs to **Cluster {cluster}**", icon="✅")

            # Show cluster context
            mask = labels == cluster
            st.markdown(f"**Cluster {cluster} profile** ({mask.sum()} existing customers):")
            st.dataframe(df[mask][feat].describe().round(2), use_container_width=True)

            # Distance bar chart
            dists = result["distances"]
            fig_d, ax_d = plt.subplots(figsize=(6, 2.5))
            colors_d = ["#2ecc71" if i == cluster else "#bdc3c7"
                        for i in range(len(dists))]
            ax_d.barh(
                [f"Cluster {i}" for i in range(len(dists))],
                dists, color=colors_d, edgecolor="white",
            )
            ax_d.set_xlabel("Distance to Centroid")
            ax_d.set_title("Distance from New Customer to Each Centroid")
            ax_d.invert_yaxis()
            fig_d.tight_layout()
            st.pyplot(fig_d)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}", icon="⚠️")
            logger.exception("Prediction error")
