from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.svm import SVR, SVC

class SVMRegression(BaseModel):
    # C_range: (default: [0.1, 1, 10])
    # epsilon_range: (default: [0.01, 0.1, 1])

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(SVR, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "C" not in self.configs:
            self.hyper_parm_grid[0]["C"]=self.configs.get("C_range", [0.1, 1, 10])
        if "epsilon" not in self.configs:
            self.hyper_parm_grid[0]["epsilon"]=self.configs.get("epsilon_range", [0.01, 0.1, 1])
        return super().fit(X, y)

class SVMClassifier(BaseModel):
    # C_range: (default: [0.1, 1, 10])
    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(SVC, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "C" not in self.configs:
            self.hyper_parm_grid[0]["C"]=self.configs.get("C_range", [0.1, 1, 10])
        return super().fit(X, y)
