import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """

    # TODO

    # END TODO

    return mse


def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.001, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal

    # END TODO

    for i in range(max_iter):
        # TODO: Compute train and test MSE

        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.

        # END TODO

        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4

        # End TODO

    return W, train_mses, test_mses


if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test)

    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)

    # End TODO
