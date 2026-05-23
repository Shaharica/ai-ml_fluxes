import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path, config):

    df = pd.read_csv(path)
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]
    return X, y

def preprocess_data(X, y, config):

    X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=config.test_size,random_state=config.split_seed)
    return (
        X_train,
        X_test,
        y_train,
        y_test
    )
