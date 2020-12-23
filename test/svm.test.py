import os, sys
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

import numpy as np
from models import SVMClassifier

from sklearn.model_selection import train_test_split
import utils

if __name__ == "__main__":

    years = utils.Groups(5)

    Groups = utils.CombinedDataSelector(years).ConstructNoFilter()

    ds = utils.DataSelector(5,Groups)

    dfs = utils.DataFromSelectorW(ds,2000,5,utils.conn)
    data = dfs.constructAll()

    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]),
        np.array(data[1]), test_size=0.1, random_state=42)

    model = SVMClassifier()
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))
    print(model.best_estimator_)