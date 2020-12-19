from __future__ import annotations
import numpy as np

from . import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class LRWithPoly(BaseModel):
    def __init__(self, *model_params, **configs):
        self.initSKModel(LinearRegression, model_params, configs)
        self.polynomial = PolynomialFeatures(configs.get("power", 5))
        self.kfold = KFold(n_splits=configs.get("kfold", 5))

    def fit(self, X: np.ndarray, y: np.ndarray) -> LRWithPoly:
        if self.model is None:
            raise NotImplementedError
        X_poly = self.polynomial.fit_transform(X)
        return self.model.fit(X_poly, y)

    def evaluate(self, X_test, y_test):
        X_poly = self.polynomial.fit_transform(X_test)
        kfold_msr = list()
        for k, (train, test) in enumerate(self.kfold.split(X_poly, y_test)):
            self.model.fit(X_poly[train], y_test[train])
            test_res = self.model.predict(X_poly[test])
            kfold_msr.append(mean_squared_error(y_test[test], test_res))
        return 0, {"kfold_msr": kfold_msr}