import covariance
import mean
import calcABC
import numpy as np

class bayesCalculation:
    def __init__(self):
        self.name = "Bayesian"

    def bayesCreation(self, covarA, covarB, meanVal):

        ABCcalc = calcABC.ABC(covarA, covarB, meanVal)
        self.A = ABCcalc.calcA()
        self.B = ABCcalc.calcB()
        self.C = ABCcalc.calcC()


    def bayesTest(self, x):
        quad = np.matmul(np.matmul(self.A, x), np.transpose(x))
        linear = np.matmul(self.B, np.transpose(x))
        return quad + linear + self.C
