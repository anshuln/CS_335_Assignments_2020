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


def ordinary_least_squares(X_train, Y_train, X_test, Y_test, lr=0.03, max_iter=5000):
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
        W -= lr * X_train.T @ (X_train @ W - Y_train) / X_train.shape[0]
    ## END TODO

    return W, train_mses, test_mses


def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.003, max_iter=200):
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


def ista(X_train, Y_train, X_test, Y_test, _lambda, lr=0.001, max_iter=5000):
    [N, D] = X_train.shape
    W = np.random.randn(D, 1)
    # prevW = W.copy()
    # lr = 0.000028
    print(N, D)

    # numepochs = 2000
    train_mses = []
    test_mses = []
    for i in range(0, max_iter):
        grad = X_train.T @ (Y_train - X_train @ W)
        # print(i, np.linalg.norm(grad))
        # 	break

        W = W + 2 * (lr / X_train.shape[0]) * grad

        # for j in range(0, D):
        #     if W[j] > _lambda:
        #         W[j] -= _lambda
        #     elif W[j] < -_lambda:
        #         W[j] += _lambda
        #     else:
        #         W[j] = 0
        W_old = W.copy()
        W -= _lambda * (W_old > _lambda)
        W += _lambda * (W_old < -_lambda)
        W[(W_old <= _lambda) * (W_old >= -_lambda)] = 0
        # print(i, "loss", math.sqrt(sse(X, Y, W)))
        train_mse = mse(X_train, Y_train, W)
        test_mse = mse(X_test, Y_test, W)

        train_mses.append(train_mse)
        test_mses.append(test_mse)
        print(train_mse, test_mse)
    return W, train_mses, test_mses


def weighted_regression(X, Y, r):
    '''
	Fill up this function for problem 3.
	Use closed form expression.
	r_train is a (n,) array, where n is number of training samples
	'''

    ## TODO
    R = np.diag(r * r)

    W = (np.linalg.inv(X.T @ R @ X)) @ (X.T @ R @ Y)
    ## END TODO
    return W


if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    W, train_mses, test_mses = ista(X_train, Y_train, X_test, Y_test, _lambda=0.001)

    # plt.plot(list(W))
    # plt.show()
    exit(0)
    # exit(0)
    # W, train_mses, test_mses = ordinary_least_squares(X_train, Y_train, X_test, Y_test)
    # W_ridge, train_mses, test_mses = ridge_regression(X_train, Y_train, X_test, Y_test, 10)
    # print(train_mses[-1], test_mses[-1])
    # Plots
    plt.figure(figsize=(4, 4))
    plt.plot(train_mses)
    plt.plot(test_mses)
    plt.legend(['Train MSE', 'Test MSE'])
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.show()
