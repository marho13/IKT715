import numpy as np
import fileReader
import mean

class covariance:
        def __init__(self, numClasses, numFeatures):
            self.sum = []
            for a in range(numClasses):
                self.sum.append([])
                for b in range(numFeatures):
                    self.sum[-1].append([])
                    for c in range(numFeatures):
                        self.sum[-1][-1].append(0)
            self.epsilon = 1e-4
            # self.Xco, self.Yco = self.covariance(self.A, self.B, meanVar)

        def covarianceCalc(self, X, mean, classNum):
            X = np.array(X)
            mean = np.array(mean)

            varianceX = [0]*len(mean)

            for i in range(len(X)):
                xNum = np.array(X[i], dtype=np.float)
                meanNum = np.array(mean[0], dtype=np.float)
                subNumX = [np.subtract(xNum, meanNum)]
                varianceX += np.matmul(np.transpose(subNumX), subNumX)

            divX = len(X)-1
            varianceX += self.epsilon
            varianceX = np.divide(varianceX, divX)
            self.sum[classNum] = varianceX
