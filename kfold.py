def kfold(data, ind, foldNumber):
    outputX = []
    outputY = []
    print(len(data))
    for a in range(len(data)):
        outputX.append([])
        numIter = len(data[a]) // foldNumber
        for b in range(foldNumber):
            if b == ind:
                outputY.append(data[a][b*numIter:(b+1)*numIter])

            else:
                for c in range((b*numIter), ((b+1)*numIter), 1):
                    outputX[-1].append(data[a][c])
    return outputX, outputY
