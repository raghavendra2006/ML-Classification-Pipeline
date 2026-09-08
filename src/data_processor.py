"""
Data Processing Module
======================
Handles loading and preprocessing of the Wine classification dataset.
Ensures reproducibility via fixed random states and proper scaler fitting.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load the Wine dataset and preprocess it for model training.

    The scaler is fitted on training data ONLY to prevent data leakage,
    then used to transform both training and test sets.

    Args:
        test_size (float): Fraction of data reserved for testing (default: 0.2).
        random_state (int): Seed for reproducible train/test splits (default: 42).

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler,
                feature_names, target_names)
    """
    # 1. Load the Wine classification dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    # 2. Split into train/test with fixed random_state for determinism
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Fit scaler on training data ONLY (prevents data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, target_names
