from .baseModel import BaseModel
from .linears import LinRegWithPoly, RidgeRegCVWithPoly, LassoCVWithPoly
from .knn import KNNRegressor
from .svm import SVMRegression
from .kerneltrick import KNNGaussianKernelReg, KernelRidge
from .trees import DecTreeReg, RandomForestReg