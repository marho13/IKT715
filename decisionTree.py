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
        self.feature = 0
        self.classes = None
        self.adjacency = []
        self.parent = None

    def addChild(self, child):
        self.adjacency.append(child)

    def setParent(self, parent):
        self.parent = parent

    def setFeature(self, feature):
        self.feature = feature

    def hasChild(self):
        if self.adjacency != []:
            return True
        else:
            return False

    def setWinner(self, classy):
        self.classes = classy

class Tree:
    def __init__(self, trainingData, testingData):
        self.training = trainingData
        self.testing = testingData
        self.head = Node()

    def calculateTree(self):
        tempTraining = []
        removedIndexes = {}
        winningOrder = []
        # winnerSplit = []

        for x in range(len(self.training)):
            tempTraining.append([])
            for y in range(len(self.training[x])):
                tempTraining[-1].append([])
                for z in range(len(self.training[x][y])):
                    tempTraining[-1][-1].append(self.training[x][y][z])

        for a in range(len(tempTraining[0][0])):
            remInd, split = self.entropyCalc(tempTraining, removedIndexes)
            winningOrder.append(remInd)
            removedIndexes.update({remInd:1})

        return winningOrder

    def fit(self):
        order = self.calculateTree()
        tempNode = None
        # [print("Most important feature is: {}".format(o)) for o in order]
        for a in range(len(order)):
            if a == 0:
                self.head.setFeature(order[a])
                tempNode = self.head
            else:
                newNode = Node()
                newNode.setFeature(order[a])
                tempNode.addChild(newNode)
                newNode.setParent(tempNode)
                tempNode = newNode

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

        outy = np.quantile(features, [0.5], axis=1)
        return outy

    def entropyCalc(self, trainingData, remInd):
        featureEntropy = []
        split, num, classes = self.train(trainingData, remInd)

        for a in range(len(split)):
            if a not in remInd:
                for b in range(len(split[a])):
                    entro = [0, 0, 0, 0]
                    totEntro = [0, 0, 0, 0]
                    for c in range(len(split[a][b])):
                        for other in range(len(split[a][b])):
                            if c != other:
                                c1 = split[a][0][c] + split[a][1][c]
                                c2 = split[a][0][other] + split[a][1][other]
                                totNum = c1 + c2
                                totEntro[c] += -(c1/totNum)*math.log((c1/totNum), 2) -(c2/totNum)*math.log((c2/totNum), 2)

                                if split[a][b] != 0:
                                    bother = split[a][b][c] + split[a][b][other]
                                    ansNum = bother/totNum
                                    entro[c] += -((ansNum)*math.log(ansNum, 2))

                entro = [((totEntro[x] - entro[x])/4) for x in range(len(entro))]
                featureEntropy.append(entro)

            else:
                featureEntropy.append([0, 0, 0, 0])

        maxNum = 0.0
        maxInd = 0
        for aa in range(len(featureEntropy)):
            tempSum = sum(featureEntropy[aa])
            if tempSum > maxNum:
                maxNum = tempSum
                maxInd = aa

        # winningClass = 0
        # maxNum = 0.0
        # for classy in range(len(featureEntropy[maxInd])):
        #     if featureEntropy[maxInd][classy] > maxNum:
        #         maxNum = featureEntropy[maxNum][classy]
        #         winningClass = classy
        return maxInd, split

    def train(self, trainingData, remInd):
        split = []
        num = 0
        classes = []
        for __ in range(len(trainingData[0][0])):
            split.append([])
            classes.append([])
            for _ in range(2):
                classes[-1].append(0)
                split[-1].append([])
                for ___ in range(len(trainingData)):
                    split[-1][-1].append(0)

        for c in range(len(trainingData)):
            for ex in range(len(trainingData[c])):
                num += 1
                for feat in range(len(trainingData[c][ex])):
                    if feat not in remInd:
                        if trainingData[c][ex][feat] < 1:
                            classes[feat][0] += 1
                            split[feat][0][c] += 1

                        elif trainingData[c][ex][feat] >= 1:
                            split[feat][1][c] += 1
                            classes[feat][1] += 1

        return split, num, classes

    def predict(self):
        for a in range(len(self.testing)):
            print(a)

dataSet = fileReader.fileRead(filey="glass_binary.csv")

for n in range(len(dataSet)):
    np.random.shuffle(dataSet[n])
train, test = kfold.kfold(dataSet, 0, 5)

t = Tree(train, test)
t.fit()
t.predict()

