
"""
tuning/tune_xgb.py
"""

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
import json
import os


def tune_xgb(X_train, y_train, config):

    param_dist = { 
        "n_estimators": [100,200, 400], 
        "learning_rate": [0.03, 0.1], 
        "max_depth": [3, 5], 
        "min_child_weight": [1, 3, 5], 
        "gamma": [0, 0.1], 
        "reg_alpha": [0, 0.1, 1], 
        "reg_lambda": [1, 5], 
        "subsample": [0.6, 0.8, 1.0], 
        "colsample_bytree": [0.5, 0.7, 1.0] }

    model = XGBRegressor(random_state=config.model_seed)

    search = GridSearchCV(
        model,
        param_grid=param_dist,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_

    os.makedirs(config.save_dir, exist_ok=True)
    path = f"{config.save_dir}/{config.experiment_name}_hyp.json"

    with open(path, "w") as f:
        json.dump(best_params, f, indent=4)

    print("[INFO] XGB best params:", best_params)

    return best_params

