# Mall Customer Segmentation — K-Means Clustering

**CST2216 Individual Term Project — Modularizing and Deploying ML Code**

A modular, production-style unsupervised machine learning project that segments mall customers using K-Means clustering, deployed as a Streamlit web application. https://clusteringan.streamlit.app/

---

## Project Structure

```
clustering/
├── app.py                      # Streamlit web application (entry point)
├── requirements.txt
├── README.md
├── data/
│   └── mall_customers.csv      # 200 customers × 5 attributes
├── logs/
│   └── app.log                 # Auto-generated runtime log
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralised configuration
│   ├── logger.py               # File + console logging
│   ├── data_loader.py          # CSV loading & validation
│   ├── preprocessing.py        # Encoding, feature selection, scaling, PCA
│   ├── train.py                # Elbow, silhouette, KMeans fit, persistence
│   ├── predict.py              # Single-customer cluster assignment
│   └── visualise.py            # All matplotlib/seaborn chart functions
└── tests/
    └── test_pipeline.py        # 14 pytest unit tests
```

---

## Quick Start

### 1. Open the project folder

```bash
cd clustering
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## App Pages

| Page | What it does |
|------|-------------|
| 📊 **Data Explorer** | Dataset overview, distributions, correlation heatmap, pairplot |
| 🔍 **Optimal K** | Elbow curve + silhouette score sweep over k=2…10 |
| 🤖 **Clustering** | Fit KMeans, view cluster scatter, PCA projection, silhouette diagram, feature box-plots, gender distribution, downloadable labelled CSV |
| 🔮 **Predict** | Enter a new customer's details and assign them to the nearest cluster |

The **feature set** and **k** are controlled from the sidebar — changes re-run the analysis instantly.

---

## Dataset

Mall Customers dataset — 200 customers, 5 columns.

| Column | Type | Description |
|--------|------|-------------|
| Customer_ID | int | Unique identifier |
| Gender | str | Male / Female |
| Age | int | Age in years |
| Annual_Income | int | Annual income in $1000s |
| Spending_Score | int | Mall spending score 1–100 |

No missing values.

---

## Algorithms

- **K-Means** with k-means++ initialisation
- **Elbow Method** (WCSS / inertia) for k selection
- **Silhouette Analysis** for k validation
- **PCA** for 2-D projection of high-dimensional feature sets

---

## Dependencies

- Python ≥ 3.10
- streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, pytest

