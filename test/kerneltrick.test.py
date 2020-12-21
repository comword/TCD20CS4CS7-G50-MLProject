import os, sys
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, src_dir)

import numpy as np
from models import KNNGaussianKernelReg

if __name__ == "__main__":
    X = np.linspace((1,2), (10,20), 100)
    y = np.concatenate((np.zeros(50), np.ones(50)))
    knngaussian = KNNGaussianKernelReg()
    knngaussian.fit(X, y)
    knngaussian.evaluate(X, y)