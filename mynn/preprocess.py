import numpy as np

def filter_by_class(X, y, class_num):
    X = X[np.isin(y, range(class_num))]
    y = y[np.isin(y, range(class_num))]
    return X, y

def train_validation_split(X_train, y_train, train_num, validation_num):
    X_val = X_train[: validation_num]
    y_val = y_train[: validation_num]
    X_train = X_train[validation_num : validation_num + train_num]
    y_train = y_train[validation_num : validation_num + train_num]
    print("Training set class distribution:", np.unique(y_train, return_counts=True))
    print("Validation set class distribution:", np.unique(y_val, return_counts=True))
    return X_train, y_train, X_val, y_val

def standardlization(X):
    # normalize from [0, 255] to [0, 1]
    X = X.astype('float32') / X.max()
    return X
