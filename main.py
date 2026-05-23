
"""
main.py

Main pipeline controller
"""

from config import get_config_from_cli

from data.loader import load_dataframe, load_data
from preprocessing.filters import apply_filters
from preprocessing.target_transform import TargetTransformer
from preprocessing.split_scale import split_and_scale

from models.ann_model import ANNModel
from models.rf_model import RFModel
from models.xgb_model import XGBModel
from tuning.ann_tune import tune_ann_bayes
from tuning.rf_tune import tune_rf
from tuning.xgb_tune import tune_xgb
from utils.plotting import results_plot
from utils.load_hyperparams import load_hyperparams
from utils.feature_importance import (compute_feature_importance,save_feature_importance,plot_feature_importance)

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
# -------------------------------
def run_pipeline(config):

    # -------- Load data --------
    if config.data == 'csat':
        df = load_dataframe('~/IIT_Roorkee/Work/arctic/Ranichauri/Project/data/processed_ranichauri.xlsx') 
    else:
        df = load_dataframe('~/IIT_Roorkee/Work/arctic/Ranichauri/data/irgasen_ranichauri.xlsx')        
    X, y = load_data(df, config)

    # -------- Filter --------
    X, y = apply_filters(X, y, df, config)

    # -------- Target transform --------
    transformer = TargetTransformer(config)
    y = transformer.transform(y)

    # -------- Split + scale --------
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, config)
    
        # -------- Model selection --------
    if config.model == "ANN":
        ModelClass = ANNModel
        tune_fn = tune_ann_bayes
    
    elif config.model == "RF":
        ModelClass = RFModel
        tune_fn = tune_rf
    
    elif config.model == "XGB":
        ModelClass = XGBModel
        tune_fn = tune_xgb
    
    else:
        raise ValueError("Invalid model")

    # -------- Mode: TUNE -in,-------
    if config.mode == "tune":
        tune_fn(X_train, y_train, config)
        return

    # -------- Mode: TRAIN --------

    if config.mode == "train":
        
        params = load_hyperparams(config)

        model = ModelClass(config)
        model.train(X_train, y_train, params)
        model.save()
        os.makedirs(f"{config.save_dir}/stats", exist_ok=True)
        if scaler is not None:
            scaler_path = f"{config.save_dir}/stats/{config.experiment_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)

    
        return

    # -------- Mode: PREDICT --------
    if config.mode == "predict":
        model = ModelClass(config)
        model.load()

        y_pred = model.predict(X_test)

        # Inverse transform
        y_pred = transformer.inverse_transform(y_pred)
        y_test = transformer.inverse_transform(y_test)

        # Metrics
        metrics = results_plot(y_pred, y_test, config)
        print("[RESULTS]", metrics)
        

        # fi_df = compute_feature_importance(
        # model.model if hasattr(model, "model") else model,
        # X_test,
        # y_test,
        # config)

        # print("\nFeature Importance:\n", fi_df)
    
        # save_feature_importance(fi_df, config)
        # plot_feature_importance(fi_df, config)
    
        return metrics,y_pred,y_test

# -------------------------------
if __name__ == "__main__":
    config = get_config_from_cli()
    run_pipeline(config)

