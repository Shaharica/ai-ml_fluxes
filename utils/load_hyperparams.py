
import json
import os

def load_hyperparams(config):
    path = f"{config.save_dir}/Hyperparameters/{config.experiment_name}_hyp.json"
   
    if not os.path.exists(path):
        print("No hyperparameters file found → using defaults")
        return {} 
    with open(path, "r") as f:
        params = json.load(f)
        print("Loaded hyperparameters")

    return params

