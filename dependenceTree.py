import fileReader
import kfold
import math
import numpy as np


class Node:
    def __init__(self):
        self.adjacency = []
        self.parent = None

    def addChild(self, child):
        self.adjacency.append(child)

    def setParent(self, parent):
        self.parent = parent

class Tree:
    def __init__(self, trainingData, testingData):
        self.training = trainingData
        self.testing = testingData
        self.tempTraining, self.tempTesting = None, None


    def calculateTree(self, meanVal, covarVal):
        pass
        #for each feature, calculate the mean for each of them
        #and the covariance (or percentiles)
        #
        #Find the one that splits the data the most
        #given this, take the one that splits it the most and so on

    def buildTree(self):
        pass

    def calcQuantiles(self, featureNum, classA, classB):
        features = []
        for _ in range(featureNum):
            features.append([])
        for a in range(len(self.training)):
            if a == classA:
                for b in range(len(self.training[a])):
                    for c in range(len(self.training[a][b])):
                        if c != 7:
                            features[c].append(self.training[a][b][c])
                        else:
                            features[c].append(self.training[a][b][c] + 1e-04)
            elif a == classB:
                for b in range(len(self.training[a])):
                    for c in range(len(self.training[a][b])):
                        if c != 7:
                            features[c].append(self.training[a][b][c])
                        else:
                            features[c].append(self.training[a][b][c] + 1e-04)


        outy = np.quantile(features, 0.5, axis=1)
        return outy

    def entropyCalc(self, featureNum):
        featureEntropy = []
        split, num, classes = self.train(featureNum, 0, 1)
        for a in range(len(split)):
            entro = 0
            for b in range(len(split[a])):
                if classes[a][b] != 0:
                    ansNum = classes[a][b]/num
                    entro += -((ansNum)*math.log(((classes[a][b])/num), 2))
            featureEntropy.append(totEntro-entro)
            #

        return featureEntropy



    def train(self, featureNum, classA, classB):
        quantiles = self.calcQuantiles(featureNum, classA, classB)
        print(quantiles)
        split = []
        num = 0
        classes = []
        testedClassPairs = {}
        for __ in range(len(self.training[0][0])):
            split.append([])
            classes.append([])
            for _ in range(2):
                classes[-1].append(0)
                split[-1].append([])
                for ___ in range(len(self.training)):
                    split[-1][-1].append(0)

        #Run across a collection of all classes
        for c in range(len(self.training)):
            if c == classA or c == classB:
                for ex in range(len(self.training[c])):
                    num += 1
                    for feat in range(len(self.training[c][ex])):
                        if self.training[c][ex][feat] < quantiles[feat]:
                            self.tempTraining[c][ex][feat] = 0
                            classes[feat][0] += 1
                            split[feat][0][c] += 1

                        else:
                            self.tempTraining[c][ex][feat] = 1
                            split[feat][1][c] += 1
                            classes[feat][1] += 1

        for classy in range(len(self.testing)):
            if classy == classA or classy == classB:
                for example in range(len(self.testing[classy])):
                    for feature in range(len(self.testing[classy][example])):
                        if self.testing[classy][example][feature] < quantiles[feature]:
                            self.tempTesting[classy][example][feature] = 0

                        else:
                            self.tempTesting[classy][example][feature] = 1

        return split, num, classes



    def informationGain(self):
        pass

dataSet = fileReader.fileRead(filey="glass.data")

for n in range(len(dataSet)):
    np.random.shuffle(dataSet[n])
train, test = kfold.kfold(dataSet, 0, 5)

t = Tree(train, test)
t.entropyCalc(len(train[0][0]))

