import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# -------------------------------
def compute_feature_importance(model, X, y, config):
    if config.target == 'L':
        feature_names = ["q8","t0","q20","t20","u20","rib_0_20"]
    else:
        feature_names=["u20","t20","t0","rib_0_20"]
        
    if config.model == "ANN":
        print("[INFO] Using permutation importance (ANN)")

        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=config.split_seed,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )

        importances = result.importances_mean

    elif config.model in ["RF", "XGB"]:
        print(f"[INFO] Using built-in importance ({config.model})")

        importances = model.feature_importances_

    else:
        raise ValueError("Unsupported model")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df


# -------------------------------
def save_feature_importance(df, config):

    os.makedirs(config.save_dir, exist_ok=True)

    path = f"{config.save_dir}/{config.experiment_name}_feature_importance.csv"
    df.to_csv(path, index=False)

    print(f"[INFO] Feature importance saved at: {path}")


# -------------------------------
def plot_feature_importance(df, config):

    plt.figure(figsize=(6,4))

    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.title(f"{config.experiment_name} Feature Importance")

    plt.tight_layout()

    path = f"{config.save_dir}/{config.experiment_name}_feature_importance.svg"
    plt.savefig(path)
    plt.show()

    print(f"[INFO] Plot saved at: {path}")