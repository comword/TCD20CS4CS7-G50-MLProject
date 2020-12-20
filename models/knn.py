from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import SCORERS
from sklearn.model_selection import GridSearchCV

class KNNRegressor(BaseModel):
    # n_neighbors: Number of neighbors to use
    # n_neighbors_range: the auto search range of n_neighbors (default: np.arange(1, 15))
    # kfold: folds number (default=5)

    model: GridSearchCV
    hyper_parm_grid = list()

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(KNeighborsRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.model
        if "n_neighbors" not in self.configs:
            self.hyper_parm_grid.append({
                "n_neighbors": self.configs.get("n_neighbors_range", np.arange(1, 15))
                })
            self.model = GridSearchCV(self.model, self.hyper_parm_grid,
                cv=self.configs.get("kfold", 5), scoring='neg_mean_squared_error')
        return self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        assert self.model
        for scorer in ["neg_mean_squared_error",
                       "neg_root_mean_squared_error", "r2"]:
            self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
        return self.metrics["val_r2"], self.metrics