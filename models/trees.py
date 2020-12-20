from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import SCORERS

class DecTreeReg(BaseModel):
    model: DecisionTreeRegressor

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(DecisionTreeRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.model != None
        return self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        assert self.model != None
        for scorer in ["neg_mean_squared_error",
                       "neg_root_mean_squared_error", "r2"]:
            self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
        return self.metrics["val_neg_mean_squared_error"], self.metrics

class RandomForestReg(DecTreeReg):
    model: RandomForestRegressor

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RandomForestRegressor, model_params, configs)


