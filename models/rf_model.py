
"""
models/rf_model.py
"""

import joblib
import os
from sklearn.ensemble import RandomForestRegressor


class RFModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    def _build_model(self, params):

        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            max_features=params.get("max_features",0.75),
            max_samples=params.get("max_samples",0.7),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
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

