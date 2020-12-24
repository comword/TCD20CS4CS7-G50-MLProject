from .baseModel import BaseModel
from .linears import LinRegWithPoly, RidgeRegCVWithPoly, LassoCVWithPoly, LogisticWithPoly, RidgeClsWithPoly
from .knn import KNNRegressor, KNNClassifier
from .svm import SVMClassifier
from .kerneltrick import KNNGaussianKernelReg, KernelRidge, KNNGaussianKernelCls
from .trees import DecTreeReg, RandomForestReg, DecTreeCls, RandomForestCls
from .baseline import DummyRegressor, DummyClassifier