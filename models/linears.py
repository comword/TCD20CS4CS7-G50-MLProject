from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
class LinRegWithPoly(BaseModel):
    # hyper-paramters:
    # degree: the degree of the polynomial
    # degree_range: the auto search range of the degree of the polynomial (default: np.arange(1, 3))
    # if degree is set, the degree_range will be omitted
    # kfold: folds number (default=5)

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', LinearRegression(**kwargs))])

        self.initSKModel(get_pipeline, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "degree" not in self.configs:
           self.hyper_parm_grid[0]["poly__degree"] = self.configs.get("degree_range", np.arange(1, 3))
        return super().fit(X, y)

    def name(self) -> str:
        return "Linear Regression"

from sklearn.linear_model import RidgeCV, LassoCV
class RidgeRegCVWithPoly(LinRegWithPoly):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, alphas=[1e-2, 1e-1, 1], **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', RidgeCV(alphas=alphas, **kwargs))])
        self.initSKModel(get_pipeline, model_params, configs)

    def name(self) -> str:
        return "Ridge Regression"

class LassoCVWithPoly(LinRegWithPoly):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, alphas=np.logspace(-2, 1, 10), **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', LassoCV(alphas=alphas, **kwargs))])
        self.initSKModel(get_pipeline, model_params, configs)

    def name(self) -> str:
        return "Lasso Regression"

# Classifications

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

class LogisticWithPoly(LinRegWithPoly):
    # C: Each of the values in Cs describes the inverse of regularization strength
    # C_range: np.logspace(-1, 1, 10)
    # penalty: 'l1', 'l2'
    # penalty_range: ['l1', 'l2']

    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, C=1, penalty='l2', max_iter=100, solver='saga', **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', LogisticRegression(solver=solver,
                                C=C, penalty=penalty, max_iter=max_iter, **kwargs))])

        self.initSKModel(get_pipeline, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "C" not in self.configs:
           self.hyper_parm_grid[0]["model__C"] = self.configs.get("C_range", np.logspace(-1, 2, 10))
        if "penalty" not in self.configs:
           self.hyper_parm_grid[0]["model__penalty"] = self.configs.get("penalty_range", ['l1', 'l2'])
        return super().fit(X, y)

    def name(self) -> str:
        return "Logistic Regression"

class RidgeClsWithPoly(LinRegWithPoly):
    # alphas: Array of alpha values to try. Regularization strength; default: [0.1, 1.0, 10.0]
    is_classification = True

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, alphas=[0.1, 1.0, 10.0], **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', RidgeClassifierCV(alphas=alphas, **kwargs))])

        self.initSKModel(get_pipeline, model_params, configs)
    
    def name(self) -> str:
        return "Ridge Classification"