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
        self.configs = configs
        raise NotImplementedError

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
        assert self.model
        if len(self.hyper_parm_grid[0]) != 0:
            self.model = GridSearchCV(self.model, self.hyper_parm_grid,
                cv=self.configs.get("kfold", 5), 
                scoring=self.configs.get("scoring", 'neg_mean_squared_error'))
        return self.model.fit(X, y)

    def evaluate(self, X_test: np.ndarray,
        y_test: np.ndarray) -> tuple[float, dict[str, float]]:

        assert self.model
        if not self.is_classification:
            for scorer in ["neg_mean_squared_error",
                        "neg_root_mean_squared_error", "r2"]:
                self.metrics["val_"+scorer] = SCORERS[scorer](self.model, X_test, y_test)
            return self.metrics["val_r2"], self.metrics
        else:
            
            return 0, self.metrics

    def __getattr__(self, *args):
        if self.model is None:
            raise NotImplementedError
        return self.model.__getattribute__(*args)
    
    def __repr__(self) -> str:
        if self.model is not None:
            return super().__repr__() + '\nModel repr: {}'.format(self.model.__repr__())
        else:
            return super().__repr__()
