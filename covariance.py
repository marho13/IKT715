import numpy as np
import fileReader
import mean

class covariance:
        def __init__(self, numClasses, numFeatures):
            self.sum = []
            for _ in range(5):
                self.sum.append([])
                for a in range(numClasses):
                    self.sum[-1].append([])
                    for b in range(numFeatures):
                        self.sum[-1][-1].append([])
                        for c in range(numFeatures):
                            self.sum[-1][-1][-1].append(0)
            self.classNumList = [0, 0, 0, 0]


        def covarianceCalc(self, X, mean, classNum, fold):
            varianceX = []
            for a in range(len(mean[0][0])):
                varianceX.append([])
                for _ in range(len(mean[0][0])):
                    varianceX[-1].append(0)

            for i in range(len(X)):
                xNum = np.array(X[i], dtype=np.float)
                meanNum = np.array(mean[fold][classNum], dtype=np.float)
                subNumX = [np.subtract(xNum, meanNum)]
                varianceX += (np.matmul(np.transpose(subNumX), subNumX))
                self.classNumList[classNum] += 1.0

            self.sum[fold][classNum] = varianceX

        def divy(self, fold):
            for b in range(len(self.sum[fold])):
                for c in range(len(self.sum[fold][b])):
                    for d in range(len(self.sum[fold][b][c])):
                        if self.sum[fold][b][c][d] == 0.0:
                            self.sum[fold][b][c][d] = 0.0

                        else:
                            self.sum[fold][b][c][d] / (self.classNumList[b]-1.0)

