
"""
models/mlp_wrapper.py

Wrapper for Bayesian optimization (skopt)
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor


class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        layer1=8,
        layer2=0,
        layer3=0,
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.01,
        tol=1e-4,
        max_iter=10000,
        solver="lbfgs",
        random_state=42,
    ):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.tol = tol
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y):
        layers = tuple(i for i in [self.layer1, self.layer2, self.layer3] if i > 0)

        self.model = MLPRegressor(
            hidden_layer_sizes=layers,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            tol=self.tol,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

