import AntiBayesCalc
import NaiveBayesCalc
import BayesCalc
import fileReader
import kfold
import numpy as np
import mean
import covariance

# Variable initiation
sum = [0, 0, 0, 0]
num = 0

def trainer(bayes, covarA, covarB, meanVal):
    bayes.bayesCreation(covarA, covarB, meanVal)

def validate(test_data, bayes, class1, class2, data):
    global sum, num

    output = bayes.bayesTest(test_data)
    if output > 0:
        return class1
    else:
        return class2


def trainandtest(method, numClasses, test):
    global covarList, meanList, sum, num
    class1 = 0
    class2 = None

    for x in range(len(test)):
        for y in range(len(test[x])):
            for f in range(numClasses):
                b = findMethod(method)
                if f != class1:
                    class2 = f
                    trainer(b, covarList[class1], covarList[class2], [meanList[class1], meanList[class2]])

                    winner = validate(test[x][y], b, class1, class2, x)
                    class1 = winner

            if class1 == x:
                sum[class1] += 1
                num += 1
            else:
                num += 1

    totSum = 0
    for s in sum:
        totSum += 1
    print(sum, (totSum/num)*100)
    sum = [0, 0, 0, 0]
    num = 0



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


def runBayes():
    print("Calculating Bayes: ")
    for i in range(5):
        train, test = dataLoader(dataSet, i)

        b = BayesCalc.bayesCalculation()
        trainandtest("bayes", len(train), test)

def runNaive():
    print("Calculating Naive Bayes: ")
    for j in range(5):
        train, test = dataLoader(dataSet, j)

        nb = NaiveBayesCalc.naiveBayesCalculation()
        trainandtest(nb, len(train), test)

def runAntiBayes():
    print("Calculating Anti Bayes: ")
    for k in range(5):
        ab = AntiBayesCalc.AntiBayesianCalculation()
        train, test = dataLoader(dataSet, k)

        trainandtest(ab, len(train), test)

dataSet = fileReader.fileRead(filey="glass.data")

for n in range(len(dataSet)):
    np.random.shuffle(dataSet[n])

m = mean.mean(4, 8)
c = covariance.covariance(4, 8)
for a in range(len(dataSet)):
    m.meanCalc(dataSet[a], a)
    c.covarianceCalc(dataSet[a], m.sum[a], a)

meanList = m.sum
covarList = c.sum

#
runBayes()
# print()
# print()
# runNaive(x, y)
#
# print()
# print()
#
# runAntiBayes()
# anti = AntiBayesCalc.antiBayesCalculation()