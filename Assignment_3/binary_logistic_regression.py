import numpy as np
import argparse
from utils import *


class BinaryLogisticRegression:
    def __init__(self, D):
        """
        D - number of features
        """
        self.D = D
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        X - numpy array of shape (N, D)
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        z = X @ self.weights
        sigmas = 1 / (1 + np.exp(-z))
        return sigmas > 0.5
        # END TODO

    def train(self, X, Y, lr=0.5, max_iter=10000):
        for _ in range(max_iter):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.

            z = X @ self.weights
            sigmas = 1 / (1 + np.exp(-z))
            y_diff = sigmas - Y
            grad = X.T @ y_diff
            self.weights -= (lr / X.shape[0]) * grad

            # END TODO

            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4

            norm = np.linalg.norm(grad)
            if norm < 1e-4:
                print("converged")
                break

            # End TODO

    def accuracy(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        accuracy = ((preds == Y).sum()) / len(preds)
        return accuracy

    def f1_score(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        # TODO: calculate F1 score for predictions preds and true labels Y
        tp = ((Y == 1.0) * (preds == Y)).sum()
        fn = ((Y == 1.0)*(preds != Y)).sum()
        fp = ((Y == 0.0) * (preds != Y)).sum()
        recall = tp/(tp + fn)
        precision = tp/(tp + fp)
        f1 = (2*recall*precision)/(recall + precision)
        return f1
        # End TODO


if __name__ == '__main__':

    X, Y = load_data('data/songs.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    D = X_train.shape[1]

    lr = BinaryLogisticRegression(D)
    lr.train(X_train, Y_train)
    preds = lr.predict(X_test)
    acc = lr.accuracy(preds, Y_test)
    f1 = lr.f1_score(preds, Y_test)
    print(f'Test Accuracy: {acc}')
    print(f'Test F1 Score: {f1}')
