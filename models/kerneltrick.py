from __future__ import annotations
import numpy as np

from . import BaseModel
from .knn import KNNRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge as SKKernelRidge
from sklearn.model_selection import GridSearchCV

def gaussian_kernel_builder(gamma):
    def gaussian_kernel(distances):
        weights = np.exp(-1*gamma*(distances**2))
        return weights/np.sum(weights)
    return gaussian_kernel

class KNeighborsGaussianRegressor(KNeighborsRegressor):
    def __init__(self, n_neighbors=5, *, gamma=10, weights=None,
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
              n_neighbors=n_neighbors,
              algorithm=algorithm,
              leaf_size=leaf_size, metric=metric, p=p,
              metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self.gamma = gamma
        self.weights = weights
    
    def predict(self, X):
        if(self.weights is None):
            self.weights = gaussian_kernel_builder(self.gamma)
        return super().predict(X)

class KNNGaussianKernelReg(KNNRegressor):
    # n_neighbors: Number of neighbors to use
    # n_neighbors_range: the auto search range of n_neighbors (default: np.arange(1, 15))
    # gamma: the gamma value for gaussian_kernel
    # gamma_range: the auto search range of gamma (default: np.logspace(2, -1, 10))
    # kfold: folds number (default=5)
    model: GridSearchCV
    hyper_parm_grid = [dict()]

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(KNeighborsGaussianRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.model
        if "n_neighbors" not in self.configs:
            self.hyper_parm_grid[0]["n_neighbors"] = self.configs.get("n_neighbors_range", np.arange(1, 15))
        if "gamma" not in self.configs:
            self.hyper_parm_grid[0]["gamma"] = self.configs.get("gamma_range", np.logspace(1.8, -1, 10))
        if len(self.hyper_parm_grid[0]) != 0:
            self.model = GridSearchCV(self.model, self.hyper_parm_grid,
                cv=self.configs.get("kfold", 5), scoring='neg_mean_squared_error')
        return self.model.fit(X, y)

    
class KernelRidge(BaseModel):
    model: GridSearchCV
    hyper_parm_grid = [dict()]

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(SKKernelRidge, model_params, configs)

    