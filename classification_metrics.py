import numpy as np

def confusion_matrix(y_true, y_pred):
    classes = list(set(y_true))
    X = np.zeros((len(classes), len(classes)))
    for i in range(len(y_true)):
        X[classes.index(y_true[i]), classes.index(y_pred[i])] += 1
    return classes, X

def accuracy(X):
    return np.sum(np.diag(X)) / np.sum(X)

def precision(X):
    return np.diag(X) / np.sum(X, axis=0)

def recall(X):
    return np.diag(X) / np.sum(X, axis=1)
