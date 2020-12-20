from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.svm import SVR
from sklearn.metrics import SCORERS
from sklearn.model_selection import GridSearchCV

class SVMRegression(BaseModel):
    # The free parameters in the model are C and epsilon.
    # C_range: (default: [0.1, 1, 10])
    # epsilon_range: (default: [0.01, 0.1, 1])
    hyper_parm_grid = [dict()]

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(SVR, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.model
        if "C" not in self.configs:
            self.hyper_parm_grid[0]["C"]=self.configs.get("C_range", [0.1, 1, 10])
        if "epsilon" not in self.configs:
            self.hyper_parm_grid[0]["epsilon"]=self.configs.get("epsilon_range", [0.01, 0.1, 1])
        if len(self.hyper_parm_grid[0]) != 0:
            self.model = GridSearchCV(self.model, self.hyper_parm_grid,
                cv=self.configs.get("kfold", 5), scoring='neg_mean_squared_error')
        return self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        assert self.model
        for scorer in ["neg_mean_squared_error",
                       "neg_root_mean_squared_error", "r2"]:
            self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
        return self.metrics["val_neg_mean_squared_error"], self.metrics
