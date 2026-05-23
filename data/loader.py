
"""
data/loader.py

Handles:
- Loading dataset
- Selecting features
- Selecting target (H / Tau)
"""

import pandas as pd


# 👉 Define your feature set here (EDIT as needed)
# FEATURE_COLUMNS = ["u27","t27","t0","rib_0_27"]
# FEATURE_COLUMNS = ["u20","t20","q20","u8","t8","q8","t0","rib_8_20"]


def load_dataframe(path=None):

    if path is None:
        raise ValueError("Please provide dataset path")
    df = pd.read_excel(path)
    # df = pd.read_excel('~/IIT_Roorkee/Work/arctic/Ranichauri/data/irgasen_ranichauri.xlsx', engine='openpyxl')
    print(f"[INFO] Loaded data: {df.shape}")

    return df


def load_data(df, config):
    
# FEATURE_COLUMNS = ["u27","t27","t0","rib_0_27"]
# FEATURE_COLUMNS = ["u20","t20","q20","u8","t8","q8","t0","rib_8_20"]

    print(config.data)
    # -------- Target Selection --------
    if config.target == "H":
        if config.data == "irg":
            FEATURE_COLUMNS = ["u20","t20","t0","rib_0_20"]
            target_col = "H_kin"
        else:
            FEATURE_COLUMNS = ["u27","t27","t0","rib_0_27"]
            target_col = "H"
        
    elif config.target == "TAU":
        if config.data == "IRG":
            FEATURE_COLUMNS = ["u20","t20","t0","rib_0_20"]
            target_col = "Tau"
        else:
            FEATURE_COLUMNS = ["u27","t27","t0","rib_0_27"]
            target_col = "Tau"
            
    elif config.target == "L":
        FEATURE_COLUMNS  = ["q8","t0","q20","t20","u20","rib_0_20"]
        target_col = "LE"
    else:
        raise ValueError(f"Invalid target: {config.target}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    y = df[target_col].copy().rename(config.target)
    
    # -------- Feature Selection --------
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing columns in dataframe: {missing_features}")
    
    X = df[FEATURE_COLUMNS].copy()
    
    # -------- Rename Features --------
    X = X.rename(columns={
    "u20": "u2",
    "u27": "u2",
    "t20": "t2",
    "t27": "t2",
    "q20": "q2",
    "q8":  "q1",
    "t0":  "t1",
    "rib_0_20": "rib",
    "rib_0_27": "rib"
})

    print(f"[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target shape: {y.shape}")

    return X, y

