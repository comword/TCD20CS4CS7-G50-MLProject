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
    # gamma_range: the auto search range of gamma (default: np.logspace(1.8, -1, 10))
    # kfold: folds number (default=5)

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(KNeighborsGaussianRegressor, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "n_neighbors" not in self.configs:
            self.hyper_parm_grid[0]["n_neighbors"] = self.configs.get("n_neighbors_range", np.arange(1, 15))
        if "gamma" not in self.configs:
            self.hyper_parm_grid[0]["gamma"] = self.configs.get("gamma_range", np.logspace(1.8, -1, 10))
        return super().fit(X, y)

class KernelRidge(BaseModel):
    # alpha: Regularization strength
    # alpha_range: default: np.logspace(-3, 0, 10)
    # kernel: string e.g. 'linear', 'polynomial', 'sigmoid', 'rbf'
    # more kernel see: https://scikit-learn.org/stable/modules/metrics.html
    # kernel_range: default: ['rbf', 'laplacian']
    # gamma: the gamma value for the kernel
    # gamma_range: the auto search range of gamma (default: np.logspace(0.1, -1, 10)
    # kfold: folds number (default=5)

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.initSKModel(SKKernelRidge, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "alpha" not in self.configs:
            self.hyper_parm_grid[0]["alpha"] = self.configs.get("alpha_range", 
                np.logspace(-3, 0, 10))
        if "kernel" not in self.configs:
            self.hyper_parm_grid[0]["kernel"] = self.configs.get("kernel_range", 
                ['rbf', 'laplacian'])
        if "gamma" not in self.configs:
            self.hyper_parm_grid[0]["gamma"] = self.configs.get("gamma_range", 
                np.logspace(0.1, -1, 10))
        return super().fit(X, y)

    