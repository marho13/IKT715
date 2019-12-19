import numpy as np
import pandas as pd
import math
from sklearn import metrics
import os


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


def cross_validation(df_in, fold):
    # To make the train and test representative, we need to make sure that we get some samples from each in each fold
    classes = df_in['Type'].unique()
    folds = []
    temp1 = [0, 0, 0, 0, 0, 0]
    for i in range(fold):
        df_temp1 = pd.DataFrame()
        for cl in range(len(classes)):
            nr_class = df_in[df_in['Type'] == classes[cl]]
            chunk = math.floor(len(nr_class) / fold)
            df_temp2 = df_in[df_in['Type'] == classes[cl]]
            df_temp1 = df_temp1.append(df_temp2.iloc[temp1[cl]:(temp1[cl] + chunk)])
            temp1[cl] += chunk
        folds.append(df_temp1)
    return folds


def make_floored(array_in):
    unique = set(array_in)
    for un in unique:
        array_in = np.where(array_in == un, un - 1, array_in)
    return array_in


if __name__ == "__main__":
    df = pd.read_csv('glass_binary.csv', delimiter=',')
    df = df[(df['Type'] == 1) | (df['Type'] == 2) | (df['Type'] == 3) | (df['Type'] == 7)]
    unique_classes = df['Type'].unique()
    print('Possible classes:\n', unique_classes)
    print('Possible columns:\n', list(df.columns))
    cross_val_folds = cross_validation(df, 5)

    selected_features = ['BinaryRI', 'BinaryNa', 'BinaryMg']
    print('Selected features:\n', selected_features)

    accuracy_list = []

    for a in range(len(cross_val_folds)):
        temp = cross_val_folds.copy()
        test_set = cross_val_folds[a]
        del temp[a]

        train_set = pd.DataFrame()
        for f in temp:
            train_set = train_set.append(f)

        # print('Length of train:', len(train_set))
        # print('Length of test:', len(test_set))

        clf = DecisionTreeClassifier(max_depth=6)
        x_in = train_set[selected_features].values
        y_in = train_set['Type'].values
        y_in = make_floored(y_in)

        # print(x_in)
        # print(y_in)
        clf.fit(x_in, y_in)

        y_test = test_set['Type'].values
        y_test = make_floored(y_test)

        predicted = clf.predict(test_set[selected_features].values)
        accuracy_list.append(metrics.accuracy_score(y_test, predicted) * 100)

    print('Mean decision tree:\n', np.mean(accuracy_list), '%')
    print('Fold Accuracy:\n', accuracy_list)

