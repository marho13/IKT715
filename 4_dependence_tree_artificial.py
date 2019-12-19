from itertools import combinations
import pandas as pd
import math
import random
import numpy as np

# THIS IS FOR THE BINARY ARTIFICIAL DATA SET


def calculate_weight(df_in):
    features = df_in.columns
    combos = list(combinations(features, 2))
    # print('Possible edges:\n', combos)
    weight_list = []
    edge_list = []
    for combo in combos:
        comb_a = list(combo)[0]
        comb_b = list(combo)[1]
        df_temp = df_in[list(combo)]

        tot = len(df_temp)

        # Calculate p(a = 0), p(a = 1), p(b = 0), p(b = 1)
        pa_0 = len(df_temp[df_temp[comb_a] == 0]) / tot
        pa_1 = len(df_temp[df_temp[comb_a] == 1]) / tot
        pb_0 = len(df_temp[df_temp[comb_b] == 0]) / tot
        pb_1 = len(df_temp[df_temp[comb_b] == 1]) / tot

        # Calculate p(a = 1, b = 0), p(a = 0, b = 0), p(a = 0, b = 1), p(a = 1, b = 1)
        pab_10 = len(df_temp[(df_temp[comb_a] == 1) & (df_temp[comb_b] == 0)]) / tot
        pab_00 = len(df_temp[(df_temp[comb_a] == 0) & (df_temp[comb_b] == 0)]) / tot
        pab_01 = len(df_temp[(df_temp[comb_a] == 0) & (df_temp[comb_b] == 1)]) / tot
        pab_11 = len(df_temp[(df_temp[comb_a] == 1) & (df_temp[comb_b] == 1)]) / tot

        part1 = pab_10 * math.log(pab_10 / (pa_1 * pb_0))
        part2 = pab_00 * math.log(pab_00 / (pa_0 * pb_0))
        part3 = pab_01 * math.log(pab_01 / (pa_0 * pb_1))
        part4 = pab_11 * math.log(pab_11 / (pa_1 * pb_1))

        weight = part1 + part2 + part3 + part4
        edge_list.append(list(combo))
        weight_list.append(weight)
    return edge_list, weight_list


# Class to represent a graph
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, -1 * w])

        # A utility function to find set of an element i

    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

        # A function that does union of two sets of x and y

    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

            # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self):

        result = []  # This will store the resultant MST

        i = 0  # An index variable, used for sorted edges
        e = 0  # An index variable, used for result[]

        # Step 1:  Sort all the edges in non-decreasing
        # order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

            # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge does't cause cycle,
            # include it in result and increment the index
            # of result for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
                # Else discard the edge

        # print the contents of result[] to display the built MST
        print("Following are the edges in the constructed MST:")
        for u, v, weight in result:
            # print str(u) + " -- " + str(v) + " == " + str(weight)
            print("{} -- {} == {}".format(u, v, weight))

        return result


def classify(x_in, dp_a, dp_b, df_train_a, df_train_b):
    prob_a = 1
    prob_b = 1
    for d in range(len(x_in) - 1):
        dep_on = dp_a[d + 1]
        curr = d + 1
        df_temp = df_train_a[(df_train_a[curr] == x_in[curr]) & (df_train_a[dep_on] == x_in[dep_on])]
        temp_prob = len(df_temp) / len(df_train_a)
        prob_a *= temp_prob

    # print('Found probability of class a', prob_a)

    for d in range(len(x_in) - 1):
        dep_on = dp_b[d + 1]
        curr = d + 1
        df_temp = df_train_b[(df_train_b[curr] == x_in[curr]) & (df_train_b[dep_on] == x_in[dep_on])]
        temp_prob = len(df_temp) / len(df_train_b)
        prob_b *= temp_prob

    # print('Found probability of class b', prob_b)

    if prob_a > prob_b:
        return True
    elif prob_b > prob_a:
        return False
    else:
        print('The classes have equal probability... The class is selected based on coin flip.')
        rnd = random.random()
        if rnd <= 0.5:
            return True
        else:
            return False


def cross_validation(df_in, fold):
    # To make the train and test representative, we need to make sure that we get some samples from each in each fold
    classes = df_in['class'].unique()
    folds = []
    temp1 = [0, 0, 0, 0, 0, 0]
    for m in range(fold):
        df_temp1 = pd.DataFrame()
        for cl in range(len(classes)):
            nr_class = df_in[df_in['class'] == classes[cl]]
            chunk = math.floor(len(nr_class) / fold)
            df_temp2 = df_in[df_in['class'] == classes[cl]]
            df_temp1 = df_temp1.append(df_temp2.iloc[temp1[cl]:(temp1[cl] + chunk)])
            temp1[cl] += chunk
        folds.append(df_temp1)
    return folds


# TODO: Classify based on the found tree


df = pd.read_csv('datasets/artificial_binary.csv', delimiter=',')

selected_features = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10']
print('Selected features:\n', selected_features)

df_test = pd.DataFrame()

dfs = []
dependencies = []


poss_classes = df['class'].unique()
for cl in poss_classes:
    temp_data = df[df['class'] == cl].iloc[:9800]
    df_test = df_test.append(df[df['class'] == cl].iloc[9800:10000])
    print('length training', len(temp_data))
    print('length test', len(df[df['class'] == cl].iloc[9800:10000]))
    temp_data = temp_data[selected_features]
    dfs.append(temp_data)
    temp_data.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    temp_edges, temp_weights = calculate_weight(temp_data)
    g = Graph(10)
    for j in range(len(temp_edges)):
        # print(weights1[j])
        g.addEdge(temp_edges[j][0], temp_edges[j][1], temp_weights[j])
    res = g.KruskalMST()
    dependency = {}
    for i in range(10):
        for j in range(len(res)):
            if res[j][1] == i:
                dependency[i] = res[j][0]
    dependencies.append(dependency)

result = []
x_values = df_test[selected_features].values
counter = 0
nrx = 0
result_random = []
for x in x_values:
    choice = random.choice(poss_classes)
    result_random.append(choice)
    nrx += 1
    print('x number:', nrx)
    temp = poss_classes
    while len(temp) != 1:
        idx_a = list(poss_classes).index(temp[-1])
        idx_b = list(poss_classes).index(temp[-2])
        ans = classify(x, dependencies[idx_a], dependencies[idx_b], dfs[idx_a], dfs[idx_b])
        if ans:
            temp = np.delete(temp, -2)
        else:
            temp = np.delete(temp, -1)
    result.append(temp[0])
for i in range(len(result)):
    if result[i] == df_test.iloc[i]['class']:
        counter += 1

counter_random = 0
for i in range(len(result_random)):
    if result_random[i] == df_test.iloc[i]['class']:
        counter_random += 1

print('Accuracy achieved by dependency tree:\n', counter / len(df_test) * 100)
print('Accuracy achieved by random classifier:\n', counter_random / len(df_test) * 100)


