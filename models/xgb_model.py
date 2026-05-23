
"""
models/xgb_model.py
"""

import joblib
import os
from xgboost import XGBRegressor


class XGBModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    def _build_model(self, params):

        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.6),
            min_child_weight=params.get("min_child_weight", 5),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_lambda=params.get("reg_lambda", 1),
            reg_alpha=params.get("reg_alpha",0),
            gamma=params.get("gamma",0),
            random_state=self.config.model_seed,
            n_jobs=-1,
        )
        return model

    def train(self, X_train, y_train, params):
        self.model = self._build_model(params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        os.makedirs(f"{self.config.save_dir}/params", exist_ok=True)
        path = f"{self.config.save_dir}/params/{self.config.experiment_name}_seed{self.config.model_seed}.pkl"
        joblib.dump(self.model, path)

    def load(self):
        path = f"{self.config.save_dir}/params/{self.config.experiment_name}_seed{self.config.model_seed}.pkl"
        self.model = joblib.load(path)

