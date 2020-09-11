import numpy as np
import argparse
from utils import *


def get_data(dataset, num_train_samples=-1):
    datasets = ['D1', 'D2']
    assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
    X_train = np.loadtxt(f'data/{dataset}/training_data')
    Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
    X_test = np.loadtxt(f'data/{dataset}/test_data')
    Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

    return X_train[:num_train_samples, :], Y_train[:num_train_samples], X_test, Y_test


def load_data1(file):
    '''
    Given a file, this function returns X, the regression features
    and Y, the output

    Args:
    filename - is a csv file with the format

    feature1,feature2, ... featureN,y
    0.12,0.31,1.33, ... ,5.32

    Returns:
    X - numpy array of shape (number of samples, number of features)
    Y - numpy array of shape (number of samples, 1)
    '''

    data = np.loadtxt(file, delimiter=',', skiprows=1)
    X = data[:, :-1]
    Y = data[:, -1:]

    return X, Y


def sigma(s):
    return 1 / (1 + np.exp(-s))


class LR:
    def __init__(self, C, D):
        """
        C - number of classes
        D - number of features
        """
        self.C = C
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        x - numpy array of shape (D,)
        """
        # TODO: Return predicted class for x
        # return np.argmax(self.weights @ (x.reshape(-1, 1)))
        sigmas = sigma(X @ self.weights)
        return sigmas > 0.45
        # END TODO

    def train(self, X, Y, lr=1, max_iter=10000):
        for _ in range(max_iter):
            sigmas = sigma(X @ self.weights)
            ydiff = Y - sigmas
            grad = X.T @ ydiff
            self.weights += (lr / X.shape[0]) * grad
            norm = np.linalg.norm(grad)
            # print("grad: ", norm)
            if norm < 1e-4:
                print("converged")
                break

    def eval(self, X, Y):
        preds = self.predict(X)
        accuracy = ((preds == np.around(Y)).sum()) / len(preds)
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to train on')
    args = parser.parse_args()

    num_train_samples = args.num_samples

    # X_train, Y_train, X_test, Y_test = get_data('D2', num_train_samples)
    X, Y = load_data2('data/songs.csv')
    X, Y = preprocess(X, Y)
    print(Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    # print(X_train.shape, Y_train.shape)

    C = max(np.max(Y_train), np.max(Y_test)) + 1
    D = X_train.shape[1]

    lr = LR(C, D)
    lr.train(X_train, Y_train)
    acc = lr.eval(X_test, Y_test)
    print(np.sum(Y_test))
    print(f'Test Accuracy: {acc}')
