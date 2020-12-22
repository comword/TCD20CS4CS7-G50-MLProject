import os, sys
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

import numpy as np
from sklearn.model_selection import train_test_split
import utils
from utils import DataSelector, DataFromSelector, conn, DataFromSelectorW

from models import *

def TrainingPipeline(daArray, conn, model):
    cds = utils.CombinedDataSelector(daArray)
    for s in cds.Construct():
        (selected,itemi) = s
        ds = DataSelector(5,selected)
        dfs = DataFromSelector(ds,1980,25,conn)
        datas = dfs.constructAll()
        (X, y) = datas
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        modelc = model.getNewConsolitedModel()
        modelc.fit(X_train,y_train)
        Score = modelc.evaluate(X_test,y_test)[0]
        print(itemi.note, Score, modelc.name())

class SingleModel:
    def __init__(self,model):
        self.model = model
    def getNewConsolitedModel(self):
        return self.model
    def name(self):
        return self.model.name()

class ConsolitedModel:
    def __init__(self, models):
        self.models = models
    def fit(self,X,y):
        scores = 1000000
        for modeli in self.models:
            model = modeli()
            model.fit(X,y)
            scoresw = model.evaluate(X,y)[0]
            if scoresw < scores:
                scores = scoresw
                self.modelselected = model
    def evaluate(self,X,y):
        return self.modelselected.evaluate(X,y)

    def name(self):
        return self.modelselected.__class__.__name__

from utils import *
ds = DataSelector(5,["NY.GDP.PCAP.PP.KD","SI.POV.LMIC.GP","SI.DST.10TH.10","SI.DST.50MD"])

dfs = DataFromSelectorW(ds,2015,1,conn)

years = utils.Groups(5)

TrainingPipeline(years,utils.conn, SingleModel(ConsolitedModel([RandomForestReg,DecTreeReg,LinRegWithPoly, RidgeRegCVWithPoly, LassoCVWithPoly,KNNGaussianKernelReg, KNNRegressor])))