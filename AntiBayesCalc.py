import numpy as np


class AntiBayesianCalculation:
    def __init__(self):
        self.name = "AntiBayesian"

    def calculateQuantiles(self, mean, covarA, covarB):
        c0q1 = mean[0] - covarA/(np.sqrt(2*np.pi))
        c1q1 = mean[1] - covarB/(np.sqrt(2*np.pi))
        c0q2 = mean[0] + covarA/(np.sqrt(2*np.pi))
        c1q2 = mean[1] + covarB/(np.sqrt(2*np.pi))

        return c0q1, c1q1, c0q2, c1q2

    def bayesCreation(self, covarA, covarB, mean):
        c0q1, c0q2, c1q1, c1q2 = self.calculateQuantiles(mean, covarA, covarB)
        self.c0q1, self.c0q2, self.c1q1, self.c1q2 = c0q1, c0q2, c1q1, c1q2


    def bayesTest(self, data):

        num0, num1  = 0, 0
        for a in range(len(self.c0q1)):
            if (self.c0q1[a][a]) < (self.c1q2[a][a]):
                x = abs(data[a]-(self.c0q1[a][a]))
                y = abs(data[a]-(self.c1q2[a][a]))

                if x > y:
                    num0 += 1
                else:
                    num1 += 1

            else:
                x = abs(data[a] - (self.c1q1[a][a]))
                y = abs(data[a] - (self.c0q2[a][a]))

                if x > y:
                    num0 += 1
                else:
                    num1 += 1

        if num0 > num1:
            return 1

        else:
            return -1