def fileRead(filey):

    data = open(filey, mode="r").read()
    data = data.split("\n")
    files = [d.split(",") for d in data]
    files = [f[1:] for f in files]
    del files[-1]
    fuckmeList = [0, 1, 2, 6]
    dataSet = [[], [], [], [], [], [], []]

    for x in files:
        num = int(x[-1])-1
        dataSet[num].append(x[:-4])
        dataSet[num][-1] = [float(d) for d in dataSet[num][-1]]

    for b in range(len(dataSet)-1, -1, -1):
        if b not in fuckmeList:
            del dataSet[b]

    return dataSet
