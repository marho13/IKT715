import numpy as np
import math
class ABC:
    def __init__(self, covarA, covarB, mean):
        self.covarA = covarA
        self.covarB = covarB
        self.mean = mean
        self.covarAInv = np.linalg.pinv(self.covarA)
        self.covarBInv = np.linalg.pinv(self.covarB)
        self.MeanAT, self.MeanBT = None, None

    def calcA(self):
        A = self.covarBInv - self.covarAInv
        return A

    def calcB(self):
        self.MeanBT = np.transpose(self.mean[1])
        self.MeanAT = np.transpose(self.mean[0])
        B = -2*np.matmul(self.covarBInv, self.MeanBT) + 2*np.matmul(self.covarAInv, self.MeanAT)

        return B

    def calcC(self):
        lnA = 0.0
        lnB = 0.0
        for a in range(len(self.covarB)):
            for b in range(len(self.covarB[a])):
                A = self.covarAInv[a][b]
                B = self.covarBInv[a][b]
                if abs(A) > 0:
                    lnA += math.log(abs(A), 2)
                if abs(B) > 0:
                    lnB += math.log(abs(B), 2)

        lnSum = lnB-lnA

        Bsum = np.matmul(np.matmul(self.covarBInv, self.mean[1]), self.MeanBT)
        Asum = np.matmul(np.matmul(self.covarAInv, self.mean[0]), self.MeanAT)
        C = lnSum + Bsum - Asum

        return C
