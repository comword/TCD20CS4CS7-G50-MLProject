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
        self.configs = configs

        def get_pipeline(degree=3, **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', LinearRegression(**kwargs))])

        self.initSKModel(get_pipeline, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if "degree" not in self.configs:
           self.hyper_parm_grid[0]["poly__degree"] = self.configs.get("degree_range", np.arange(1, 3))
        return super().fit(X, y)

from sklearn.linear_model import RidgeCV, LassoCV
class RidgeRegCVWithPoly(LinRegWithPoly):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, alphas=[1e-2, 1e-1, 1], **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', RidgeCV(alphas=alphas, **kwargs))])
        self.initSKModel(get_pipeline, model_params, configs)

class LassoCVWithPoly(LinRegWithPoly):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=3, alphas=np.logspace(-2, 1, 10), **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', LassoCV(alphas=alphas, **kwargs))])
        self.initSKModel(get_pipeline, model_params, configs)

# Classifications

from sklearn.linear_model import LogisticRegressionCV

class LogisticWithPoly(BaseModel):
    is_classification = True

    def __init__(self, *model_params, **configs):
        self.configs = configs

        def get_pipeline(degree=3, **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('model', LogisticRegressionCV(**kwargs))])

        self.initSKModel(get_pipeline, model_params, configs)