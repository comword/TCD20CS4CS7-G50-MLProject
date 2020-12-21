from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

class KNNRegressor(BaseModel):
    # n_neighbors: Number of neighbors to use
    # n_neighbors_range: the auto search range of n_neighbors (default: np.arange(1, 15))
    # kfold: folds number (default=5)

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(KNeighborsRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "n_neighbors" not in self.configs:
            self.hyper_parm_grid[0]["n_neighbors"] = self.configs.get("n_neighbors_range", np.arange(1, 15))
        return super().fit(X, y)

class KNNClassifier(KNNRegressor):
    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(KNeighborsClassifier, model_params, configs)