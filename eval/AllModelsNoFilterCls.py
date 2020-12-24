import os, sys

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

from utils import Groups, TrainingPipelineW, conn, SingleModel, ConsolitedModel

from models import *

years = Groups(5)

TrainingPipelineW(years, conn, SingleModel(ConsolitedModel(
    [RidgeClsWithPoly, KNNClassifier, LogisticWithPoly, SVMClassifier, DecTreeCls, RandomForestCls, DummyClassifier])))
