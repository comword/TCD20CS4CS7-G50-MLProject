from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import SCORERS

class DecTreeReg(BaseModel):
    # max_depth
    # max_depth_range: np.logspace(0, 2, 10).astype(int)
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(DecisionTreeRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "max_depth" not in self.configs:
            self.hyper_parm_grid[0]["max_depth"] = self.configs.get("max_depth_range", np.logspace(0, 2, 10).astype(int))
        return super().fit(X, y)

class RandomForestReg(BaseModel):
    # n_estimators
    # n_estimators_range: np.logspace(1.5, 2.5, 5).astype(int)

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RandomForestRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "n_estimators" not in self.configs:
            self.hyper_parm_grid[0]["n_estimators"] = self.configs.get("n_estimators_range", np.logspace(1.5, 2.5, 5).astype(int))
        return super().fit(X, y)

class DecTreeCls(DecTreeReg):
    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(DecisionTreeClassifier, model_params, configs)

class RandomForestCls(RandomForestReg):
    model: RandomForestClassifier

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RandomForestClassifier, model_params, configs)
