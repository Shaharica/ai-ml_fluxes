import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def feature_imp(model,X_test,y_test,model_name,config):
   
    feature_names = X_test.columns.tolist()

    # ANN
    if model_name == "ANN":
        print("[INFO] Using permutation importance (ANN)")
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=config.split_seed,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        importances = result.importances_mean

    # RF / XGB
    elif model_name in ["RF", "XGB"]:
        print(f"[INFO] Using built-in importance ({model_name})")
        importances = model.feature_importances_
    else:
        raise ValueError("Unsupported model")
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(
        "importance",
        ascending=False
    )
    return df
# -------------------------------
def save_feature_importance(df,model_name,config):
    path = (
        f"{config.save_dir}/"
        f"{config.experiment_name}_{model_name}_feature_importance.csv"
    )

    df.to_csv(  path,index=False)
    print(f"[INFO] Feature importance saved at: {path}")

def plot_feature_importance( df,model_name,config):

    plt.figure(figsize=(6,4))

    # sort for cleaner plotting
    df = df.sort_values(
        "importance",
        ascending=True
    )

    plt.barh(
        df["feature"],
        df["importance"]
    )

    plt.xlabel("Importance")

    plt.title(
        f"{config.experiment_name} "
        f"{model_name} Feature Importance"
    )

    plt.tight_layout()

    path = (
        f"{config.plot_dir}/"
        f"{config.experiment_name}_{model_name}_feature_importance.png"
    )
    plt.savefig(path)
    print(f"[INFO] Plot saved at: {path}")