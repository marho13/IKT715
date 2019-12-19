import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import math
import seaborn as sn
import matplotlib.pyplot as plt
import random
# Optimal Bayesian classifier with dependent variables
# Right now, the optimal gets accuracy for some cases smaller than the own made naive bayes classifier
# Hence, probably, there is something wrong with the script
random.seed(42)

def old_make_covariance_matrix(df_in, types_in):
    means = df_in.groupby('Type').mean()
    cov_matrixes = []
    for i in range(len(types_in)):
        df_temp1 = df_in[df_in['Type'] == types_in[i]]
        df_temp2 = df_temp1[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
        data_points_c = df_temp2.values
        mean_temp = means.values[i]
        print('mean_temp', mean_temp)
        sum_temp = 0
        for x in data_points_c:
            # print('x', x)
            # print('mean temp', mean_temp)
            nr1 = (x - mean_temp)
            # print(nr1)
            nr2 = np.ndarray.transpose(x - mean_temp)
            # print(nr2)
            temp3 = np.matmul(nr1, nr2)
            sum_temp += temp3
        sum_temp = sum_temp / (len(data_points_c - 1))
        cov_matrixes.append(sum_temp)
    return cov_matrixes


def new_make_covariance_matrix(df_in, types_in):
    means = df_in.groupby('Type').mean()
    cov_matrixes = []
    for i in range(len(types_in)):
        df_temp1 = df_in[df_in['Type'] == types_in[i]]
        df_temp2 = df_temp1[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
        data_points_c = df_temp2.values
        mean_temp = means.values[i]
        sum_temp = 0
        for x in data_points_c:
            x_mat = np.matrix(x)
            m_mat = np.matrix(mean_temp)
            nr1 = (x_mat - m_mat)
            nr2 = nr1.transpose()
            temp3 = np.matmul(nr2, nr1)  # In paper from John AA^T, but that gives a vector, I want a kxk matrix.
            sum_temp += temp3
        sum_temp = sum_temp / (len(data_points_c - 1))
        cov_matrixes.append(sum_temp)
    return cov_matrixes


def old_find_c(x_in, cov_a, cov_b, mean_a, mean_b):
    c_a = np.matrix(cov_a)
    c_b = np.matrix(cov_b)
    values = np.matrix(x_in)
    m_a = np.matrix(mean_a)
    m_b = np.matrix(mean_b)
    c = np.log(np.absolute(c_b)) - np.log(np.absolute(c_a)) + \
        np.matmul(np.matmul(np.linalg.pinv(c_b), (values - m_b)), (np.transpose(values - m_b))) - \
        np.matmul(np.matmul(np.linalg.pinv(c_a), (values - m_a)), (np.transpose(values - m_a)))
    if np.array(c)[0][0] > 0:
        return True
    else:
        return False


def new_find_c(x_in, cov_a, cov_b, mean_a, mean_b):
    c_a = np.matrix(cov_a)
    c_b = np.matrix(cov_b)
    values = np.matrix(x_in)
    m_a = np.matrix(mean_a)
    m_b = np.matrix(mean_b)

    eig_values_b = np.linalg.eigvals(c_b)
    temp_b = []
    for n in range(len(eig_values_b)):
        if eig_values_b[n] > 1e-12:
            temp_b.append(eig_values_b[n])
    pseudo_determinant_b = np.product(eig_values_b)

    eig_values_a = np.linalg.eigvals(c_a)
    temp_a = []
    for n in range(len(eig_values_a)):
        if eig_values_a[n] > 1e-12:
            temp_a.append(eig_values_a[n])
    pseudo_determinant_a = np.product(temp_a)

    nr2_b1 = (values - m_b)
    nr2_b2 = nr2_b1.transpose()
    c_b_m1 = np.linalg.pinv(c_b)

    nr2_a1 = (values - m_a)
    nr2_a2 = nr2_a1.transpose()
    c_a_m1 = np.linalg.pinv(c_a)

    temp3 = np.matmul(nr2_b1, np.matrix(c_b_m1))
    temp4 = np.matmul(nr2_a1, np.matrix(c_a_m1))

    temp5 = np.matmul(np.matrix(temp3), nr2_b2)
    temp6 = np.matmul(np.matrix(temp4), nr2_a2)

    # Sometimes the determinants are zero. There are no log for 0 values.
    # If this occurs, we will calculate c as follows:
    if (pseudo_determinant_b != 0.0) and (pseudo_determinant_a != 0.0):
        print('NONE zero', pseudo_determinant_b, pseudo_determinant_a)
        c = np.log(pseudo_determinant_b) - np.log(pseudo_determinant_a) + temp5 - temp6
    elif pseudo_determinant_b != 0.0:
        print('A zero', pseudo_determinant_b, pseudo_determinant_a)
        c = np.log(pseudo_determinant_b) + temp5 - temp6
    elif pseudo_determinant_a != 0.0:
        print('B zero', pseudo_determinant_b, np.log(pseudo_determinant_a))
        c = - np.log(pseudo_determinant_a) + temp5 - temp6
        print('c', c)
    else:
        print('BOTH zero', pseudo_determinant_b, pseudo_determinant_a)
        c = temp5 - temp6

    if np.array(c)[0][0] > 0:
        # x belongs to class a
        return True
    else:
        # x belongs to class b
        return False


def new_classify(x_in, means_in, cov_matrixes_in, types_in):
    result = []
    x_values = x_in[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values
    counter = 0
    for x in x_values:
        temp = types_in
        while len(temp) != 1:
            idx_a = list(types_in).index(temp[-1])
            idx_b = list(types_in).index(temp[-2])
            m_a = means_in[idx_a]
            m_b = means_in[idx_b]
            c_a = cov_matrixes_in[idx_a]
            c_b = cov_matrixes_in[idx_b]
            ans = new_find_c(x, c_a, c_b, m_a, m_b)
            if ans:
                temp = np.delete(temp, -2)
            else:
                temp = np.delete(temp, -1)
        result.append(temp[0])
    for i in range(len(result)):
        if result[i] == x_in.iloc[i]['Type']:
            counter += 1
    return counter / len(x_in) * 100


def new_classify2(x_in, means_in, cov_matrixes_in, types_in):
    result = []
    x_values = x_in[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values
    for x in x_values:
        temp = types_in
        while len(temp) != 1:
            idx_a = list(types_in).index(temp[-1])
            idx_b = list(types_in).index(temp[-2])
            m_a = means_in[idx_a]
            m_b = means_in[idx_b]
            c_a = cov_matrixes_in[idx_a]
            c_b = cov_matrixes_in[idx_b]
            ans = new_find_c(x, c_a, c_b, m_a, m_b)
            if ans:
                temp = np.delete(temp, -2)
            else:
                temp = np.delete(temp, -1)
        result.append(temp[0])
    return result


def cross_validation(df_in, fold):
    # To make the train and test representative, we need to make sure that we get some samples from each in each fold
    classes = df_in['Type'].unique()
    folds = []
    temp = [0, 0, 0, 0, 0, 0]
    for i in range(fold):
        df_temp1 = pd.DataFrame()
        for cl in range(len(classes)):
            nr_class = df_in[df_in['Type'] == classes[cl]]
            chunk = math.floor(len(nr_class) / fold)
            df_temp2 = df_in[df_in['Type'] == classes[cl]]
            df_temp1 = df_temp1.append(df_temp2.iloc[temp[cl]:(temp[cl] + chunk)])
            temp[cl] += chunk
        folds.append(df_temp1)
    return folds


# Collect the data and divide it into one training data set and one testing data set
df = pd.read_csv('glass.csv', delimiter=',')
print(df.columns)
unique_classes = df['Type'].unique()

nr_in_each_class = df.groupby('Type').count()
list_of_nr = []
for nr in range(len(nr_in_each_class)):
    list_of_nr.append(nr_in_each_class.iloc[nr].RI)

print('Number of examples in each class\n', list_of_nr)

# Test that the cross validation works:
cross_val = cross_validation(df, 5)[1]
cross_val_folds = cross_validation(df, 5)
print('Test cross validation\n', cross_val)
print(cross_val.Type.unique())
print(cross_val.groupby('Type').count())


tot_acc = []
naive_tot = []
conf_mat_ob = []
# Test the classifier on the folds (with 5 fold cross validation)
for fo in range(len(cross_val_folds)):
    temp11 = cross_val_folds.copy()
    print('temp bef:', len(temp11))
    test_set = cross_val_folds[fo]
    del temp11[fo]
    print('temp af:', len(temp11))
    train_set = pd.DataFrame()
    for f in temp11:
        train_set = train_set.append(f)
    print('Len train:', len(train_set))
    print('Len test:', len(test_set))
    mean_3 = train_set.groupby('Type').mean()
    cov_m_3 = new_make_covariance_matrix(train_set, unique_classes)
    print('test set:\n', test_set.head())
    ob_res = new_classify2(test_set, mean_3.values, cov_m_3, unique_classes)
    ob_acc = metrics.accuracy_score(test_set['Type'].values, np.array(ob_res))
    print('Accuracy achieved by new optimal bayes classifier: {} %'.format(ob_acc))
    tot_acc.append(ob_acc * 100)

    conf_mat_ob.append(metrics.confusion_matrix(test_set['Type'].values, np.array(ob_res)))

    model3 = GaussianNB()
    model3.fit(train_set[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']], train_set['Type'])
    predicted3 = model3.predict(test_set[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']])

    print('For comparison, the inbuilt function achieved:',
          metrics.accuracy_score(test_set['Type'], predicted3) * 100, '%')
    naive_tot.append(metrics.accuracy_score(test_set['Type'], predicted3) * 100)

print('Total achieved optimal Bayesian:', tot_acc)
print('Total achieved Naive:', naive_tot)

print('Mean optimal Bayesian:', np.mean(tot_acc))
print('Mean Naive:', np.mean(naive_tot))