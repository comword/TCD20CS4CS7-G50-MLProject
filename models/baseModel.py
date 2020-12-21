from __future__ import annotations
from abc import abstractmethod
import inspect
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS

class BaseModel:
    """
    Base class for all models
    """
    model = None
    configs: dict
    metrics: dict[str, float] = dict()

    hyper_parm_grid: list[dict] = [dict()]
    is_classification: bool = False

    @abstractmethod
    def __init__(self, *model_params, **configs):
        # model_params are used to initialise the sklearn model 
        # configs are used to configure options outside the sklearn model
        self.model = None
        self.configs = configs
        self.metrics = dict()
        self.hyper_parm_grid = [dict()]

    def initSKModel(self, skModel, args, kwargs):
        # forward configs to the sklearn model
        if inspect.isclass(skModel):
            sig = inspect.signature(skModel.__init__)
        else:
            sig = inspect.signature(skModel)
        sk_params = sig.parameters.keys()
        forward_args = dict()
        for arg in kwargs.keys():
            if arg in sk_params:
                forward_args[arg] = kwargs[arg]
        self.model = skModel(*args, **forward_args)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert self.model is not None
        if len(self.hyper_parm_grid[0]) != 0:
            if not self.is_classification:
                scoring=self.configs.get("scoring", 'neg_mean_squared_error')
            else:
                scoring=self.configs.get("scoring", 'accuracy')
            self.model = GridSearchCV(self.model, self.hyper_parm_grid,
                cv=self.configs.get("kfold", 5), 
                scoring=scoring)
        fit_result = self.model.fit(X, y)
        if isinstance(self.model, GridSearchCV):
            # boundary warning
            model_params = self.model.best_estimator_.get_params()
            for params_dict in self.hyper_parm_grid:
                model_params_in_grid = { key: model_params[key] for key in params_dict.keys() }
                for (k, v) in model_params_in_grid.items():
                    if v == params_dict[k][0] or v == params_dict[k][-1]:
                        print("Hyperparameter search is hitting boundary: {} {}".format(k, v))
        return fit_result

    def evaluate(self, X_test: np.ndarray,
        y_test: np.ndarray) -> tuple[float, dict[str, float]]:

        assert self.model is not None
        if not self.is_classification:
            for scorer in ["neg_mean_squared_error",
                        "neg_root_mean_squared_error", "r2"]:
                self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
            return self.metrics["val_r2"], self.metrics
        else:
            for scorer in ["accuracy",
                        "roc_auc", "average_precision"]:
                self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
            return self.metrics["val_roc_auc"], self.metrics

    def __getattr__(self, *args):
        if self.model is None:
            raise NotImplementedError
        return self.model.__getattribute__(*args)
    
    def __repr__(self) -> str:
        if self.model is not None:
            return super().__repr__() + '\nModel repr: {}'.format(self.model.__repr__())
        else:
            return super().__repr__()
