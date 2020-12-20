from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import SCORERS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
class LinRegWithPoly(BaseModel):
    # hyper-paramters:
    # degree: the degree of the polynomial
    # degree_range: the auto search range of the degree of the polynomial (default: np.arange(1, 6))
    # if degree is set, the degree_range will be omitted
    # kfold: folds number (default=5)

    hyper_parm_grid = list()

    def __init__(self, *model_params, **configs):
        self.configs = configs

        def get_pipeline(degree=5, **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('reg', LinearRegression(**kwargs))])

        self.initSKModel(get_pipeline, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.model
        if "degree" not in self.configs:
            # grid search
            self.hyper_parm_grid.append({
                "poly__degree": self.configs.get("degree_range", np.arange(1, 6))
                })
            self.model = GridSearchCV(self.model, self.hyper_parm_grid,
                cv=self.configs.get("kfold", 5), scoring='neg_mean_squared_error')
        return self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        assert self.model
        for scorer in ["neg_mean_squared_error",
                       "neg_root_mean_squared_error", "r2"]:
            self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
        return self.metrics["val_neg_mean_squared_error"], self.metrics

from sklearn.linear_model import RidgeCV, LassoCV
class RidgeRegCVWithPoly(LinRegWithPoly):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=5, **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('reg', RidgeCV(**kwargs))])
        self.initSKModel(get_pipeline, model_params, configs)

class LassoCVWithPoly(LinRegWithPoly):
    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        def get_pipeline(degree=5, **kwargs):
            return Pipeline([('poly', PolynomialFeatures(degree)),
                             ('reg', LassoCV(**kwargs))])
        self.initSKModel(get_pipeline, model_params, configs)

if __name__ == "__main__":
    lasso = LassoCVWithPoly(degree=3)
    print(lasso.named_steps['poly'].degree)