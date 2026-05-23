
"""
tuning/tune_ann_bayes.py

Phase-wise Bayesian tuning for ANN (MLPRegressor)
"""

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence, plot_objective

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from models.mlp_wrapper import MLPWrapper


# -------------------------------
# Phase 1: architecture
# -------------------------------
def get_phase1_space():
    return {
        "layer1": Integer(0, 10),
        "layer2": Integer(0, 10),
        "layer3": Integer(0, 10),
        "activation": Categorical(["relu", "tanh","logistic"]),
    }


# -------------------------------
# Phase 2: regularization
# -------------------------------
def get_phase2_space():
    return {
        "alpha": Real(1e-8, 1e-2),
    }


# -------------------------------
def run_bayes(X, y, space, config, phase, fixed_params=None, n_iter=60):

    print(f"[INFO] Running {phase}...")

    base_estimator = MLPWrapper(**(fixed_params or {}))

    search = BayesSearchCV(
        estimator=base_estimator,
        search_spaces=space,
        n_iter=n_iter,
        n_points=3,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=config.split_seed,
        refit=True,
    )

    search.fit(X, y)

    # -------- Save outputs --------
    os.makedirs(config.save_dir, exist_ok=True)

    # Convergence plot
    plot_convergence(search.optimizer_results_[0])
    plt.savefig(f"{config.save_dir}/{config.experiment_name}irg_conv.svg")
    plt.close()

    # Objective plot
    plot_objective(search.optimizer_results_[0])
    plt.savefig(f"{config.save_dir}/{config.experiment_name}irg_obj.svg")
    plt.close()

    # CV results
    pd.DataFrame(search.cv_results_).to_csv(
        f"{config.save_dir}/{config.experiment_name}_irg_cv.csv",
        index=False,
    )

    return search.best_params_


# -------------------------------
def tune_ann_bayes(X_train, y_train, config, fixed_phase1=None, run_phase1=True, run_phase2=True):

    # -------- Phase 1 --------
    if fixed_phase1 is not None:
        params_p1 = fixed_phase1
        print("[INFO] Using fixed Phase 1 params:", params_p1)
    elif run_phase1:
        params_p1 = run_bayes(
            X_train,
            y_train,
            get_phase1_space(),
            config,
            phase="phase1",
            n_iter=120,)

    print("[INFO] Phase 1 best:", params_p1)

    # -------- Phase 2 --------
    if run_phase2:
        params_p2 = run_bayes(
            X_train,
            y_train,
            get_phase2_space(),
            config,
            phase="phase2",
            fixed_params=params_p1,
            n_iter=60,
        )

        print("[INFO] Phase 2 best:", params_p2)
    else:
        params_p2 = {}
        print("[INFO] Skipping Phase 2")

    # -------- Final params --------
    final_params = {
        **params_p1,
        **params_p2,
        "solver": "lbfgs",
    }

    # Save params
    path = f"{config.save_dir}/{config.experiment_name}_hyp.json"
    with open(path, "w") as f:
        json.dump(final_params, f, indent=4)

    print(f"[INFO] Final params saved at: {path}")

    return final_params

