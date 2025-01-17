import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    '''
	Compute mean squared error between predictions and true y values

	Args:
	X - numpy array of shape (n_samples, n_features)
	Y - numpy array of shape (n_samples, 1)
	W - numpy array of shape (n_features, 1)
	'''

    ## TODO
    mse = 0.5 * np.mean((X @ W - Y) ** 2)
    ## END TODO

    return mse


def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.003, max_iter=1000):
    '''
	reg - regularization parameter (lambda in Q2.1 c)
	'''
    train_mses = []
    test_mses = []

    ## TODO
    # Initialize W using using random normal
    W = np.random.randn(X_train.shape[1], 1)
    ## END TODO

    for i in range(max_iter):
        ## TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        ## END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        ## TODO: Update w and b using a single step of gradient descent
        W -= lr * (X_train.T @ (X_train @ W - Y_train) / X_train.shape[0] + 2 * reg * W)
    ## END TODO

    return W, train_mses, test_mses


def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.001, max_iter=10000):
    '''
    	reg - regularization parameter (lambda in Q2.1 c)
    	'''
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    W = np.random.randn(X_train.shape[1], 1)
    # END TODO

    for i in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.

        W_prev = W.copy()
        grad = X_train.T @ (X_train @ W - Y_train)
        W -= 2 * (lr / X_train.shape[0]) * grad
        W_old = W.copy()
        W -= _lambda * lr * (W_old > _lambda * lr)
        W += _lambda * lr * (W_old < -_lambda * lr)
        W[(W_old <= _lambda * lr) * (W_old >= -_lambda * lr)] = 0
        # END TODO

        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if np.linalg.norm(W - W_prev) < 1e-4:
            print(_lambda, "converged")
            break
        # End TODO

    return W, train_mses, test_mses


if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    lambdas = range(1, 61, 4)
    lambdas = [i / 10 for i in lambdas]
    lambdas = [0.1, 0.12, 0.16, 0.2, 0.25, 0.4, 0.5, 0.9, 2, 2.5, 6]
    # print(len(lambdas))
    # lambdas = [0.1]
    lam_test_mses = []
    lam_train_mses = []
    for lam in lambdas:
        W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, _lambda=lam)
        lam_test_mses.append(test_mses_ista[-1])
        lam_train_mses.append(train_mses_ista[-1])

    # print(lam_train_mses, lam_test_mses)
    # exit(0)

    plt.plot(lambdas, lam_train_mses)
    plt.plot(lambdas, lam_test_mses)
    plt.legend(['Train', 'Test'])
    plt.show()
    exit(0)
    W_list = [i[0] for i in W.tolist()]
    # print(W_list)
    # plt.hist(W_list, bins="auto")
    plt.scatter(range(0, len(W_list)), W_list)
    plt.show()
    exit(0)
    # exit(0)
    # W, train_mses, test_mses = ordinary_least_squares(X_train, Y_train, X_test, Y_test)
    W_ridge, train_mses, test_mses = ridge_regression(X_train, Y_train, X_test, Y_test, 10)
    # print(train_mses[-1], test_mses[-1])
    # Plots
    print(train_mses_ista[-1], test_mses_ista[-1], train_mses[-1], test_mses[-1])
    exit(0)
    plt.figure(figsize=(4, 4))
    plt.plot(train_mses_ista)
    plt.plot(test_mses_ista)
    plt.plot(train_mses)
    plt.plot(test_mses)
    plt.legend(['Train MSE ISTA', 'Test MSE ISTA', 'Train MSE Ridge', 'Test MSE Ridge'])
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.show()
