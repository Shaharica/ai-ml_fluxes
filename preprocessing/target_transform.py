
"""
preprocessing/target_transform.py

Handles:
- Forward transform (before training)
- Inverse transform (after prediction)
"""

import numpy as np
from sklearn.preprocessing import MaxAbsScaler


class TargetTransformer:
    def __init__(self, config):
        self.config = config

    def transform(self, y):
        print("[INFO] transforming")
        if self.config.target == "H":

            if self.config.stability == "unstable":
                # log(H)
                assert (y > 0).all(), "Found non-positive H in unstable regime"
                return np.log(y+1)
                

            elif self.config.stability == "stable":
                # log(-H)
                assert (y < 0).all(), "Found non-negative H in stable regime"
                return np.log(-y+1)
            else :
                return np.sign(y) * np.log1p(np.abs(y))
                

        elif self.config.target == "TAU":
            assert (y > 0).all(), "Found non-positive TAU"
            return np.log(y)+1
            # return y
        elif self.config.target == "L":
            return np.sign(y) * np.log1p(np.abs(y))
         

        else:
            raise ValueError("Invalid target")

    def inverse_transform(self, y_pred):
        """
        Convert predictions back to physical space
        """
        if self.config.target == "H":

            if self.config.stability == "unstable":
                return (np.exp(y_pred)-1)

            elif self.config.stability == "stable":
                return (-np.exp(y_pred)-1)
            else:
                return np.sign(y_pred) * (np.expm1(np.abs(y_pred)))

        elif self.config.target == "TAU":
            return (np.exp(y_pred-1))
            # return np.abs(y_pred)
            
        elif self.config.target == "L":
            return np.sign(y_pred) * np.expm1(np.abs(y_pred))
                 
        else:
            raise ValueError("Invalid target")

