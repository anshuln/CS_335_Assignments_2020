import numpy as np
import matplotlib.pyplot as plt
from utils import load_data1, split_data

def mse(X, Y, w, b):
    '''
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, 1)
    Y - numpy array of shape (n_samples, 1)
    w - a float
    b - a float
    '''

    ## TODO

    ## END TODO

    return mse

def ordinary_least_squares(w, b, X_train, Y_train, X_test, Y_test, lr, max_iter):
    train_mses = []
    test_mses = []

    for i in range(max_iter):
        ## TODO: Compute train and test MSE

        ## END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        ## TODO: Update w and b using a single step of gradient descent
        
        ## END TODO

    return w, b, train_mses, test_mses

# Load and split data
X, Y = load_data1('data1.csv')
X_train, Y_train, X_test, Y_test = split_data(X, Y)

# Initialize w and b and other parameters
## TODO
w = 0
b = 0

max_iter = 5
lr = 0.001
## END TODO

w, b, train_mses, test_mses = ordinary_least_squares(w, b, X_train, Y_train, X_test, Y_test, lr, max_iter)

# Plots
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(train_mses)
plt.plot(test_mses)
plt.legend(['Train MSE', 'Test MSE'])
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.subplot(122)
plt.plot([-20, 50], [-20*w+b, 50*w+b], color='r')
plt.scatter(X_train, Y_train, color='b', marker='.')
plt.scatter(X_test, Y_test, color='g', marker='x')
for x, y in zip(X_test, Y_test):
    plt.plot([x, x], [w*x+b, y], color='gray', zorder=-1)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()