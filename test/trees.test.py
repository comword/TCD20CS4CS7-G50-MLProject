import os, sys
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

import numpy as np
from models import DecTreeReg, RandomForestReg

from sklearn.model_selection import train_test_split
import utils

if __name__ == "__main__":

    years = utils.Groups(5)

    Groups = utils.CombinedDataSelector(years).ConstructNoFilter()

    ds = utils.DataSelector(5,Groups)

    dfs = utils.DataFromSelector(ds,1970,35,utils.conn)

    data = dfs.constructAll()

    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]),
        np.array(data[1]), test_size=0.1, random_state=42)

    model = DecTreeReg()
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))

    model = RandomForestReg()
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))