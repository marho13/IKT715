reader = open("glass.data", mode="r").read()

newLine = reader.split("\n")
lines = []
for n in newLine:
    lines.append(n.split(","))
del lines[-1]
classes = {1:0, 2:1, 3:2, 7:3}
counter = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
dataSet = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
for line in lines:
    if int(line[-1]) in classes:
        for a in range(len(line)-1):
            if a != 0:
                indexy = int(line[-1])
                dataSet[classes[indexy]][a-1] += float(line[a])
        counter[classes[indexy]] += 1.0

for d in range(len(dataSet)):
    for e in range(len(dataSet[d])):
        dataSet[d][e] /= counter[d]

writer = open("glass_binary.csv", mode="w")
for line in lines:
    if int(line[-1]) in classes:
        for val in range(len(line)):
            if val != 0 and val != (len(line)-1):
                print(val, len(dataSet[classes[int(line[-1])]]), float(line[val]))
                print(dataSet[classes[int(line[-1])]][val-1])
                if float(line[val]) >= dataSet[classes[int(line[-1])]][val-1]:
                    writer.write("1")
                    writer.write(",")
                else:
                    writer.write("0")
                    writer.write(",")

            elif val == (len(line)-1):
                writer.write(str(classes[int(line[-1])]))

        if line != []:
            writer.write("\n")

writer.close()
