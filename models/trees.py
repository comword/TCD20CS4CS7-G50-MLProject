from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import SCORERS

class DecTreeReg(BaseModel):
    model: DecisionTreeRegressor

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(DecisionTreeRegressor, model_params, configs)

class RandomForestReg(DecTreeReg):
    model: RandomForestRegressor

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RandomForestRegressor, model_params, configs)

class DecTreeCls(BaseModel):
    model: DecisionTreeClassifier
    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(DecisionTreeClassifier, model_params, configs)

class RandomForestCls(DecTreeCls):
    model: RandomForestClassifier

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RandomForestClassifier, model_params, configs)
