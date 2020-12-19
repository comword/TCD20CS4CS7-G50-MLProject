from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
class LinRegWithPoly(BaseModel):
    # hyper-paramters:
    # power: the number of the polynomial power (default=3)
    # kfold: folds number (default=5)

    # models tools
    model: LinearRegression
    polynomial: PolynomialFeatures
    kfold: KFold

    # evaluation metrics
    metrics: dict[str, float] = dict()

    def __init__(self, *model_params, **configs):
        self.configs = configs
        self.polynomial = PolynomialFeatures(configs.get("power", 3))
        self.kfold = KFold(n_splits=configs.get("kfold", 5))
        self.initSKModel(LinearRegression, model_params, configs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegressorMixin:
        assert self.model
        X_poly = self.polynomial.fit_transform(X)
        mse = list()
        for _, (train, test) in enumerate(self.kfold.split(X_poly, y)):
            self.model.fit(X_poly[train], y[train])
            test_res = self.model.predict(X_poly[test])
            mse.append(mean_squared_error(y[test], test_res))
        mse = np.array(mse)
        self.metrics["kfold_mse"] = mse.mean()
        self.metrics["kfold_std"] = mse.std()
        return self.model.fit(X_poly, y)

    def evaluate(self, X_test, y_test):
        assert self.model
        X_poly = self.polynomial.fit_transform(X_test)
        pred = self.model.predict(X_poly)
        self.metrics["val_mse"] = mean_squared_error(y_test, pred)
        self.metrics["val_rmse"] = mean_squared_error(y_test, pred, squared=False)
        self.metrics["val_r2"] = r2_score(y_test, pred)
        return 0, self.metrics

from sklearn.linear_model import RidgeCV, LassoCV
class RidgeRegCVWithPoly(LinRegWithPoly):
    model: RidgeCV

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(RidgeCV, model_params, configs)

class LassoCVWithPoly(LinRegWithPoly):
    model: LassoCV

    def __init__(self, *model_params, **configs):
        super().__init__(*model_params, **configs)
        self.initSKModel(LassoCV, model_params, configs)