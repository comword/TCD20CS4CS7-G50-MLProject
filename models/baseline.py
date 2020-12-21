from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.dummy import DummyRegressor as SKDR, DummyClassifier as SKDC

class DummyRegressor(BaseModel):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(SKDR, model_params, configs)

class DummyClassifier(BaseModel):
    is_classification = True
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(SKDC, model_params, configs)