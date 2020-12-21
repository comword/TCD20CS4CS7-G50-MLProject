from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.svm import SVR

class SVMRegression(BaseModel):
    # The free parameters in the model are C and epsilon.
    # C_range: (default: [0.1, 1, 10])
    # epsilon_range: (default: [0.01, 0.1, 1])

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(SVR, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "C" not in self.configs:
            self.hyper_parm_grid[0]["C"]=self.configs.get("C_range", [0.1, 1, 10])
        if "epsilon" not in self.configs:
            self.hyper_parm_grid[0]["epsilon"]=self.configs.get("epsilon_range", [0.01, 0.1, 1])
        return super().fit(X, y)
