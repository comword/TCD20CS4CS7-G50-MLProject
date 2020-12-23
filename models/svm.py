from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.svm import SVC

class SVMClassifier(BaseModel):
    # C_range: (default: [0.1, 1, 10])
    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(SVC, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "C" not in self.configs:
            self.hyper_parm_grid[0]["C"]=self.configs.get("C_range", np.logspace(0, 1.5, 10))
        return super().fit(X, y)

    def name(self) -> str:
        return "SVM Classifier"
