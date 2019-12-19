import AntiBayesCalc
import NaiveBayesCalc
import BayesCalc
import fileReader
import kfold
import numpy as np
import mean
import covariance
import random

random.seed(42)

class runnerforTrainTest:
    def __init__(self, method, numClasses, test, fold, covarList, meanList):
        self.covarList = covarList
        self.meanList = meanList


    def trainer(self, bayes, covarA, covarB, meanVal):
        bayes.bayesCreation(covarA, covarB, meanVal)

    def validate(self, test_data, bayes, class1, class2):

        output = bayes.bayesTest(test_data)
        if output > 0:
            return class1
        else:
            return class2


    def trainandtest(self, method, test, fold):
        sumList = [0, 0, 0, 0]
        incorrList = [0, 0, 0, 0]
        num = 0

        for x in range(len(test)):
            for y in range(len(test[x])):
                winner = self.win(method, test[x][y], fold)

                if winner == x:
                    sumList[winner] += 1
                    num += 1
                else:
                    num += 1
                    incorrList[winner] += 1

        totSum = 0
        for s in sumList:
            totSum += s
        print(totSum, num, incorrList)
        print(sumList, (totSum/num)*100)


    def win(self, method, data, fold):
        a = findMethod(method)
        b = findMethod(method)
        self.trainer(a, self.covarList[fold][0], self.covarList[fold][1], [self.meanList[fold][0], self.meanList[fold][1]])
        self.trainer(b, self.covarList[fold][2], self.covarList[fold][3], [self.meanList[fold][2], self.meanList[fold][3]])

        winner = self.validate(data, a, 0, 1)
        winner2 = self.validate(data, b, 2, 3)
        # # print(winner, winner2)

        c = findMethod(method)
        self.trainer(c, covarList[fold][winner], covarList[fold][winner2], [meanList[fold][winner], meanList[fold][winner2]])
        totWinner = self.validate(data, c, winner, winner2)
        return totWinner



def findMethod(method):
    if method == "bayes":
        b = BayesCalc.bayesCalculation()
    elif method == "anti":
        b = AntiBayesCalc.AntiBayesianCalculation()
    else:
        b = NaiveBayesCalc.naiveBayesCalculation()
    return b


def dataLoader(dataSet, ind): #rewrite to give you all of the training and all of the testing files
    print("Fold ", ind)
    return kfold.kfold(dataSet, ind, 5)


def runBayes(covarList, meanList):
    print("Calculating Bayes: ")
    for i in range(5):
        a = runnerforTrainTest("bayes", len(train[i]), test[i], i, covarList, meanList)
        a.trainandtest("bayes", test[i], i)

def runNaive():
    print("Calculating Naive Bayes: ")
    for j in range(5):
        trainandtest("naive", len(train), test[j], j)

def runAntiBayes():
    print("Calculating Anti Bayes: ")
    for k in range(5):
        train, test = dataLoader(dataSet, k)
        trainandtest("anti", len(train), test)

dataSet = fileReader.fileRead(filey="glass.data")

for n in range(len(dataSet)):
    np.random.shuffle(dataSet[n])

m = mean.mean(len(dataSet), len(dataSet[0][0]))
c = covariance.covariance(len(dataSet), len(dataSet[0][0]))
train, test = [], []
for i in range(5):
    tempTrain, tempTest = dataLoader(dataSet, i)
    train.append(tempTrain)
    test.append(tempTest)

for a in range(len(train)):
    for b in range(len(train[a])):
        m.meanCalc(train[a][b], b, a)
        c.covarianceCalc(train[a][b], m.sumList, b, a)
    c.divy(a)

meanList = m.sumList
covarList = c.sum

# for f in range(len(covarList)):
#     for c in range(len(covarList[f])):
#         covarList[f][c] = skcov.empirical_covariance(train[f][c])
# print(skcov.empirical_covariance(train[0][0]))


#
runBayes(covarList, meanList)
# print()
# print()
# runNaive(x, y)
#
# print()
# print()
#
# runAntiBayes()
# anti = AntiBayesCalc.antiBayesCalculation()