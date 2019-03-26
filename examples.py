import scipy
import numpy as np
from numpy import *
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore', message='the matrix subclass is not the recommended way')


def classification_error(preds,L_test):
    error = 0.0
    for i in range(len(preds)):
        error += float(abs(int(preds[i])-int(L_test[i]))) / 2.0
    error /= len(preds)    
    return error

def parse(filename):
    ifile = open(filename, 'r')
    lines = ifile.readlines()
    L = []
    X = []
    for line in lines:
        line = line.split()
        L.append(int(line[0]))
        x = []
        for i in range(1, len(line)):
            coordinate = line[i]
            coordinate = coordinate.split(':')[1]
            x.append(float(coordinate))
        X.append(np.array(x))
    ifile.close()
    return X, L

def get_text_data(my_random_generator):
    filename = "text.mat"
    ratio_labeled, ratio_unlabeled, ratio_test = 0.025, 0.475, 0.5
    import scipy.io
    content = scipy.io.loadmat(filename, struct_as_record=True)
    X = content['X']
    y = content['y']
    L = mat(zeros((X.shape[0], 1)))
    for i in range(len(y)):
        L[i,0] = float(y[i])
    R = list(range(0,X.shape[0]))
    my_random_generator.shuffle(R)
    indices_l = R[:int(ratio_labeled*len(R))]
    indices_u = R[int(ratio_labeled*len(R)):int(ratio_labeled*len(R)) + int(ratio_unlabeled*len(R))]
    indices_t = R[int(ratio_labeled*len(R)) + int(ratio_unlabeled*len(R)):]
    X_train_l = X[indices_l]
    X_train_u = X[indices_u]
    X_test = X[indices_t]
    L_train_l = L[indices_l].ravel().tolist()[0]
    L_test = L[indices_t].ravel().tolist()[0]
    
    kw = {}
    kw["lambda"] = 0.00390625
    kw["lambda_Uvec"] = [1]
    print("\nSparse text data set instance")
    print("Number of labeled patterns: ", X_train_l.shape[0])
    print("Number of unlabeled patterns: ", X_train_u.shape[0])
    print("Number of test patterns: ", X_test.shape[0])
    return X_train_l, L_train_l, X_train_u, X_test, L_test, kw

def get_gaussian_data(my_random_generator):
    ratio_labeled, ratio_unlabeled, ratio_test = 0.05, 0.45, 0.5
    X, L = parse("G2C.dat")
    Z = list(zip(X,L))
    my_random_generator.shuffle(Z)
    X,L = zip(*Z) 
    X = list(X)
    L = list(L)
    X_train_l = X[:int(len(X)*ratio_labeled)]
    L_train_l = L[:int(len(X)*ratio_labeled)]
    X_train_u = X[int(len(X)*ratio_labeled):int(len(X)*ratio_labeled) + int(len(X)*ratio_unlabeled) ]
    X_test = X[int(len(X)*ratio_labeled) + int(len(X)*ratio_unlabeled):]
    L_test = L[int(len(X)*ratio_labeled) + int(len(X)*ratio_unlabeled):]
    print("\nDense gaussian data set instance")
    print("Number of labeled patterns: ", len(X_train_l))
    print("Number of unlabeled patterns: ", len(X_train_u))
    print("Number of test patterns: ", len(X_test))
    return X_train_l, L_train_l, X_train_u, X_test, L_test


def get_moons_data(my_random_generator):
    ratio_labeled, ratio_unlabeled, ratio_test = 0.005, 0.495, 0.5
    X, L = parse("moons.dat")
    Z = list(zip(X,L))
    my_random_generator.shuffle(Z)
    X,L = zip(*Z) 
    X = list(X)
    L = list(L)
    X_train_l = X[:int(len(X)*ratio_labeled)]
    L_train_l = L[:int(len(X)*ratio_labeled)]
    X_train_u = X[int(len(X)*ratio_labeled):int(len(X)*ratio_labeled) + int(len(X)*ratio_unlabeled) ]
    X_test = X[int(len(X)*ratio_labeled) + int(len(X)*ratio_unlabeled):]
    L_test = L[int(len(X)*ratio_labeled) + int(len(X)*ratio_unlabeled):]
    print("\nDense moons data set instance")
    print("Number of labeled patterns: ", len(X_train_l))
    print("Number of unlabeled patterns: ", len(X_train_u))
    print("Number of test patterns: ", len(X_test))
    return X_train_l, L_train_l, X_train_u, X_test, L_test


