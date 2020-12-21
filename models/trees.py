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

class RandomForestReg(DecTreeReg):
    model: RandomForestRegressor

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RandomForestRegressor, model_params, configs)


