import numpy as np
import argparse
from utils import *


def get_data(dataset):
    datasets = ['D1', 'D2']
    assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
    X_train = np.loadtxt(f'data/{dataset}/training_data')
    Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
    X_test = np.loadtxt(f'data/{dataset}/test_data')
    Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

    return X_train, Y_train, X_test, Y_test


def one_hot_encode(X, labels):
    '''
    Args:
    X - numpy array of shape (n_samples, 1)
    labels - list of all possible labels for current category

    Returns:
    X in one hot encoded format (numpy array of shape (n_samples, n_labels))
    '''
    X.shape = (X.shape[0], 1)
    newX = np.zeros((X.shape[0], len(labels)))
    label_encoding = {}
    for i, l in enumerate(labels):
        label_encoding[l] = i
    for i in range(X.shape[0]):
        newX[i, label_encoding[X[i, 0]]] = 1
    return newX


class LR:
    def __init__(self, C, D):
        """
        C - number of classes
        D - number of features
        """
        self.C = C
        self.D = D
        self.weights = np.random.rand(D, C)

    def softmax(self, X):
        """
        :param X: (N, D) array
        :return: softmax for given X and current weights
        """
        exps = np.exp(X @ self.weights)
        return exps/np.sum(exps, axis=1).reshape(-1, 1)

    def predict(self, X):
        """
        :param X: numpy array of shape (N, D)
        :return:
        """
        # TODO: Return predicted class for x
        softmax = self.softmax(X)
        return np.argmax(softmax, 1)
        # END TODO

    def gradient(self, X, Y):
        """
        compute gradient wrt weights
        :param X: (N, D) numpy array
        :param Y: (N, C) numpy array
        :return: gradients of softmax wrt weights
        """
        s = self.softmax(X)
        dW = X.T @ (Y - s)
        return dW/X.shape[0]

    def train(self, X, Y, lr=0.1, max_iter=1000):
        print(np.sum(self.softmax(X)))
        for i in range(max_iter):
            grad = self.gradient(X, Y)
            self.weights += lr*grad
            print(i, np.linalg.norm(grad))

    def eval(self, X, Y):
        preds = self.predict(X)
        accuracy = ((preds == np.around(Y)).sum()) / len(preds)
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Problem 4')
    parser.add_argument('--dataset', type=str, default='D1',
                        help='Dataset to train on')
    args = parser.parse_args()

    # num_train_samples = args.num_samples

    X_train, Y_train, X_test, Y_test = get_data(args.dataset)
    print(Y_train)
    Y_train = one_hot_encode(Y_train, np.unique(Y_train))
    print(X_train.shape, X_test.shape, Y_train.shape, np.max(X_train))
    # X, Y = load_data2('data/songs.csv')
    # X, Y = preprocess(X, Y)
    # print(Y)
    # X_train, Y_train, X_test, Y_test = split_data(X, Y)
    # print(X_train.shape, Y_train.shape)

    C = max(np.max(Y_train), np.max(Y_test)) + 1
    D = X_train.shape[1]

    lr = LR(C, D)
    lr.train(X_train, Y_train)
    acc = lr.eval(X_test, Y_test)
    print(np.sum(Y_test))
    print(f'Test Accuracy: {acc}')
