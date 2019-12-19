class mean:
    def __init__(self, numClasses, numFeatures):
        self.sumList = []
        for _ in range(5):
            self.sumList.append([])
            for a in range(numClasses):
                self.sumList[-1].append([])
                for b in range(numFeatures):
                    self.sumList[-1][-1].append(0.0)

    def meanCalc(self, x, classNum, fold):
        num = [0, 0, 0, 0]
        for a in x:
            num[classNum] += 1.0
            for z in range(len(a)):
                self.sumList[fold][classNum][z] += float(a[z])

        for s in range(len(self.sumList[fold])):
                self.sumList[fold][classNum][s] /= float(num[classNum])
