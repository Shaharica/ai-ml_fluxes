
"""
preprocessing/split_scale.py

Handles:
- Train/Test split
- Conditional scaling (ANN only)
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(X, y, config):
    # -------- Train/Test Split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.split_seed,
        shuffle=True,
    )

    # -------- Scaling (ONLY for ANN) --------
    if config.model == "ANN":
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    else:
        scaler = None
        print(f"[INFO] No scaling applied ({config.model})")

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler

