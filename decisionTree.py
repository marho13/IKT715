import re
import random
import fileReader
import numpy as np
import kfold
import math
import mean
import covariance

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


    def calculateTree(self, meanVal, covarVal):
        pass
        #for each feature, calculate the mean for each of them
        #and the covariance (or percentiles)
        #
        #Find the one that splits the data the most
        #given this, take the one that splits it the most and so on

    def buildTree(self):
        pass

    def calcQuantiles(self, featureNum):
        features = []
        for _ in range(featureNum):
            features.append([])
        for a in range(len(self.training)):
            for b in range(len(self.training[a])):
                for c in range(len(self.training[a][b])):
                    if c != 7:
                        features[c].append(self.training[a][b][c])
                    else:
                        features[c].append(self.training[a][b][c] + 1e-04)

        # outy = [np.percentile(feat, [0.33, 0.66]) for feat in features]
        outy = np.quantile(features, [0.33, 0.66], axis=1)
        return outy

    def entropyCalc(self, featureNum):
        featureEntropy = []
        split, num, classes = self.train(featureNum)
        f1 = len(self.training[0])/num
        f2 = len(self.training[1])/num
        f3 = len(self.training[2])/num
        f4 = len(self.training[3])/num
        totEntro = -(f1*math.log(f1, 2))-(f2*math.log(f2, 2))-(f3*math.log(f3, 2))-(f4*math.log(f4, 2))
        for a in range(len(split)):
            entro = 0
            for b in range(len(split[a])):
                if classes[a][b] != 0:
                    ansNum = classes[a][b]/num
                    entro += -((ansNum)*math.log(((classes[a][b])/num), 2))
            featureEntropy.append(totEntro-entro)
            #

        return featureEntropy



    def train(self, featureNum):
        quantiles = self.calcQuantiles(featureNum)
        print(quantiles)
        split = []
        num = 0
        classes = []
        for __ in range(len(self.training[0][0])):
            split.append([])
            classes.append([])
            for _ in range(3):
                classes[-1].append(0)
                split[-1].append([])
                for ___ in range(len(self.training)):
                    split[-1][-1].append(0)
        for c in range(len(self.training)):
            for ex in range(len(self.training[c])):
                num += 1
                for feat in range(len(self.training[c][ex])):
                    if self.training[c][ex][feat] < quantiles[0][feat]:
                        classes[feat][0] += 1
                        split[feat][0][c] += 1

                    elif self.training[c][ex][feat] > quantiles[0][feat] and self.training[c][ex][feat] < quantiles[1][feat]:
                        split[feat][1][c] += 1
                        classes[feat][1] += 1

                    else:
                        split[feat][2][c] += 1
                        classes[feat][2] += 1

        return split, num, classes

    def informationGain(self):
        pass

dataSet = fileReader.fileRead(filey="glass.data")

for n in range(len(dataSet)):
    np.random.shuffle(dataSet[n])
train, test = kfold.kfold(dataSet, 0, 5)

t = Tree(train, test)
t.entropyCalc(len(train[0][0]))