import numpy as np
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
    if (pseudo_determinant_b != 0.0) and (pseudo_determinant_b != 0.0):
        # print('NONE zero', pseudo_determinant_b, pseudo_determinant_a)
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
