
"""
tuning/tune_rf.py
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json
import os


def tune_rf(X_train, y_train, config):

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4,8,12.16, 20],
        "min_samples_split": [2, 5, 10,20],
        "min_samples_leaf": [1,4,8,10],
        
    # CRITICAL CHANGE (since only 4 features)
    "max_features": [1.0, 0.75, 0.5],        # = 4, 3, 2 features

    # Optional but useful
    "max_samples": [0.7, 0.9, 1.0]
    }

    model = RandomForestRegressor(random_state=config.model_seed)

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

    print("[INFO] RF best params:", best_params)

    return best_params
