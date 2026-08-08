"""
Load and validate the Mall Customers dataset.
"""

import pandas as pd
from src.config import DATA_PATH, CUSTOMER_ID_COL
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    
    logger.info("Loading data from: %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Dataset not found: %s", path)
        raise

    logger.info("Loaded %d rows × %d columns.", *df.shape)

    expected = {CUSTOMER_ID_COL, "Gender", "Age", "Annual_Income", "Spending_Score"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    logger.info("No missing values: %s", df.isnull().sum().sum() == 0)
    return df
