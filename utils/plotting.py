
"""
utils/plotting.py

Observed vs predicted plot with metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

def apply_physical_filter(y_true, y_pred, target, stability):

    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if target == "H":
        if stability == "stable":
            mask = y_pred > -100 
            min_val = -100
            max_val = 0
        else:  # unstable
            mask = y_pred < 500
            min_val = 0
            max_val = 500

    elif target == "TAU":
        mask = np.abs(y_pred) < 3
        min_val=0
        max_val = 3
    elif target == "L":
        mask = np.ones_like(y_pred, dtype=bool)
        min_val = -1000
        max_val= 1000


    else:
        mask = np.ones_like(y_pred, dtype=bool)

    return y_true[mask], y_pred[mask],min_val,max_val
    
def results_plot(Y_pred, Y_test, config):
    # -------- Clean NaNs --------
    mask = np.isfinite(Y_test) & np.isfinite(Y_pred)
    Y_test = np.array(Y_test)[mask]
    Y_pred = np.array(Y_pred)[mask]
    y_test, y_pred, min_val, max_val = apply_physical_filter(Y_test, Y_pred, config.target, config.stability)
    
    # -------- Metrics --------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_test, y_pred, alpha=0.6)


    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 1:1 line
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")

    # Zero lines
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    # Labels
    ax.set_title(config.experiment_name)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_aspect("equal", adjustable="box")

    # Metrics box
    textstr = (
        f'R² = {r2:.3f}\n'
        f'Corr = {corr:.3f}\n'
        f'RMSE = {rmse:.3f}\n'
        f'MAE = {mae:.3f}'
    )

    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # -------- Save --------
    os.makedirs(config.save_dir, exist_ok=True)
    path = f"{config.save_dir}/plots/{config.experiment_name}_{config.model_seed}.svg"
    plt.savefig(path)
    plt.show()
    return {
        "corr": corr,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

