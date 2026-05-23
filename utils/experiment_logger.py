
"""
utils/experiment_logger.py

Logs experiment results to CSV
"""

import pandas as pd
import os


def log_experiment(config, metrics, extra_params=None):

    log_file = f"{config.save_dir}/experiment_log.csv"

    row = {
        "model": config.model,
        "target": config.target,
        "stability": config.stability,
        "experiment": config.experiment_name,
        "split_seed": getattr(config, "split_seed", None),
        "model_seed": getattr(config, "model_seed", None),
    }

    # Add metrics
    for k, v in metrics.items():
        row[k] = float(v)
    # Add params if available
    if extra_params is not None:
        for k, v in extra_params.items():
            try:
                row[f"param_{k}"] = float(v)
            except:
                row[f"param_{k}"] = str(v)

    df_new = pd.DataFrame([row])
    # Append or create
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(log_file, index=False)

