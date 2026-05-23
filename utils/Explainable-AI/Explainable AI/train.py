from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_rf(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model


def train_ann(X_train, y_train, config):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    model.fit(
        X_train,
        y_train,
        epochs=config.ann_epochs,
        batch_size=config.ann_batch_size,
        verbose=0
    )
    return model