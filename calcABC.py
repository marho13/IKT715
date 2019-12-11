import numpy as np
import math
class ABC:
    def __init__(self, covarA, covarB, mean):
        self.covarA = covarA
        self.covarB = covarB
        self.mean = mean
        self.covarAInv, self.covarBInv = None, None
        self.MeanAT, self.MeanBT = None, None

    def calcA(self):
        self.covarAInv = np.linalg.inv(self.covarA)
        self.covarBInv = np.linalg.inv(self.covarB)
        A = self.covarBInv - self.covarAInv
        return A

    def calcB(self):
        self.MeanBT = np.transpose(self.mean[1])
        self.MeanAT = np.transpose(self.mean[0])
        B = -2*np.matmul(self.covarBInv, self.MeanBT) + 2*np.matmul(self.covarAInv, self.MeanAT)

        return B

    def calcC(self): ##Read me!
        lnA = math.log1p(np.linalg.det(self.covarA))
        lnB = math.log1p(np.linalg.det(self.covarB))

        lnSum = lnB-lnA

        Bsum = np.matmul(np.matmul(self.covarBInv, self.mean[1]), self.MeanBT)
        Asum = np.matmul(np.matmul(self.covarAInv, self.mean[0]), self.MeanAT)
        C = lnSum + Bsum - Asum

        return C
