class mean:
    def __init__(self, numClasses, numFeatures):
        self.sum = []
        for a in range(numClasses):
            self.sum.append([])
            for b in range(numFeatures):
                self.sum[-1].append(0)
        self.num = [0, 0]

    def meanCalc(self, x, classNum):
        for a in x:
            self.num[0] += 1
            for z in range(len(a)):
                self.sum[classNum][z] += float(a[z])

        for s in range(len(self.sum)):
                self.sum[classNum][s] /= self.num[0]
