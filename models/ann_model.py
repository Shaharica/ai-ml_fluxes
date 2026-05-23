import joblib
from sklearn.neural_network import MLPRegressor
import os

class ANNModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    def _build_model(self, params):
        # Build hidden layer tuple (ignore zeros)
        layers = [params.get("layer1", 0),
                  params.get("layer2", 0),
                  params.get("layer3", 0)]

        hidden_layers = tuple([l for l in layers if l > 0])

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=params.get("activation", "relu"),
            alpha=params.get("alpha", 0.0001),
            learning_rate_init=params.get("learning_rate_init", 0.001),
            max_iter=params.get("max_iter", 50000),
            solver=params.get("solver", "lbfgs"),
            tol=params.get("tol", 1e-4),
            random_state=self.config.model_seed,
        )

        return model

    def train(self, X_train, y_train, params):
        self.model = self._build_model(params)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X)

    def save(self):
        os.makedirs(f"{self.config.save_dir}/params", exist_ok=True)
        path = f"{self.config.save_dir}/params/{self.config.experiment_name}_seed{self.config.model_seed}.pkl"
        joblib.dump(self.model, path)
        
    def load(self):
        path = f"{self.config.save_dir}/params/{self.config.experiment_name}_seed{self.config.model_seed}.pkl"
        self.model = joblib.load(path)
        print(f"[INFO] Model loaded from: {path}")

