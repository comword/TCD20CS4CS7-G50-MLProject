from __future__ import annotations
from abc import abstractmethod
import inspect
import numpy as np

class BaseModel:
    """
    Base class for all models
    """
    model = None
    configs: dict
    metrics: dict[str, float] = dict()

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

    def __getattr__(self, *args):
        if self.model is None:
            raise NotImplementedError
        return self.model.__getattribute__(*args)
    
    def __repr__(self) -> str:
        if self.model is not None:
            return super().__repr__() + '\nModel repr: {}'.format(self.model.__repr__())
        else:
            return super().__repr__()

    @abstractmethod
    def evaluate(self, X_test: np.ndarray,
        y_test: np.ndarray) -> tuple[float, dict[str, float]]:
        # Return: (score, map of evaluation details)
        raise NotImplementedError
