"""
This file contains all configuration settings for project
--> Easy to fix if anything changes in the future
"""

import os

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "mall_customers.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "data")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

# Data settings
CUSTOMER_ID_COL = "Customer_ID"
GENDER_COL      = "Gender"

# Available numeric feature sets for clustering
FEATURE_SETS = {
    "Income & Spending Score":       ["Annual_Income", "Spending_Score"],
    "Age & Spending Score":          ["Age", "Spending_Score"],
    "Age & Income":                  ["Age", "Annual_Income"],
    "Age, Income & Spending Score":  ["Age", "Annual_Income", "Spending_Score"],
}

# KMeans settings
K_MIN         = 2
K_MAX         = 10          # inclusive — used for elbow & silhouette sweep
DEFAULT_K     = 5
RANDOM_STATE  = 42
KMEANS_INIT   = "k-means++"
KMEANS_N_INIT = 10

# PCA settings 
PCA_N_COMPONENTS = 2        # for 3-D feature visualisation projection

# Logging
LOG_FILE  = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = "INFO"
