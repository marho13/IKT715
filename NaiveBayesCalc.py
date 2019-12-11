import covariance
import mean
import calcABC
import numpy as np
class naiveBayesCalculation:
    def __init__(self):
        self.name = "NaiveBayesian"

    def bayesCreation(self, covarA, covarB, meanVal):
        identity = np.identity(len(meanVal[0]))
        covarA = covarA*identity
        covarB = covarB*identity

        ABCcalc = calcABC.ABC(covarA, covarB, meanVal)
        self.A = ABCcalc.calcA()
        self.B = ABCcalc.calcB()
        self.C = ABCcalc.calcC()


    def bayesTest(self, x):
        quad = np.matmul(np.matmul(self.A, x), np.transpose(x))
        linear = np.matmul(self.B, np.transpose(x))
        return quad + linear + self.C
