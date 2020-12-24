import os, sys

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

from utils import Groups, TrainingPipelineA, conn, SingleModel, ConsolitedModel

from models import *

years = Groups(5)

TrainingPipelineA(years, conn, SingleModel(ConsolitedModel(
    [RandomForestReg, DecTreeReg, LinRegWithPoly, RidgeRegCVWithPoly, LassoCVWithPoly, KNNGaussianKernelReg,
     KNNRegressor, KernelRidge, DummyRegressor])))
