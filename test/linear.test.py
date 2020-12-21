import os, sys
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

import numpy as np
from models import LinRegWithPoly, LogisticWithPoly, RidgeClsWithPoly

from sklearn.model_selection import train_test_split
import utils

if __name__ == "__main__":

    years = utils.Groups(5)

    Groups = utils.CombinedDataSelector(years).ConstructNoFilter()

    ds = utils.DataSelector(5, Groups)

    dfs = utils.DataFromSelector(ds,1970,35,utils.conn)

    data = dfs.constructAll()

    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]),
        np.array(data[1]), test_size=0.1, random_state=42)

    model = LinRegWithPoly(degree_range=[1, 2, 3])
    model.fit(X_train, y_train)
    print(model.evaluate(X_test, y_test))
    print(model.best_estimator_)

    # classification
    dfs = utils.DataFromSelectorW(ds,2000,5,utils.conn)
    data = dfs.constructAll()

    X_train, X_test, y_train, y_test = train_test_split(np.array(data[0]),
        np.array(data[1]), test_size=0.1, random_state=42)

    cls = LogisticWithPoly(degree_range=[1,2], penalty='l2', solver='saga')
    cls.fit(X_train, y_train)
    print(cls.evaluate(X_test, y_test))
    print(cls.best_estimator_)

    cls = RidgeClsWithPoly(degree_range=[1,2], alphas=np.logspace(-3, 1, 10))
    cls.fit(X_train, y_train)
    print(cls.evaluate(X_test, y_test))
    print(cls.best_estimator_)
    print(cls.best_estimator_.named_steps['model'].alpha_)

