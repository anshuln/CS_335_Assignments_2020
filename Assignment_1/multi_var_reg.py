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
	mse = np.mean((X@W-Y)**2)
	## END TODO

	return mse

def ordinary_least_squares(W, X_train, Y_train, X_test, Y_test, lr=0.00001, max_iter=200):
	train_mses = []
	test_mses = []

	for i in range(max_iter):

		## TODO: Compute train and test MSE
		train_mse = mse(X_train, Y_train, W) 
		test_mse = mse(X_test, Y_test, W)
		## END TODO

		train_mses.append(train_mse)
		test_mses.append(test_mse)

		## TODO: Update w and b using a single step of gradient descent
		W -= lr * 2 * X_train.T @ (X_train@W - Y_train)
		## END TODO

	return W, train_mses, test_mses

def ridge_regression(W, X_train, Y_train, X_test, Y_test, reg, lr=0.00001, max_iter=200):
	train_mses = []
	test_mses = []

	for i in range(max_iter):

		## TODO: Compute train and test MSE
		train_mse = mse(X_train, Y_train, W) 
		test_mse = mse(X_test, Y_test, W)
		## END TODO

		train_mses.append(train_mse)
		test_mses.append(test_mse)

		## TODO: Update w and b using a single step of gradient descent
		W -= lr * (2 * X_train.T @ (X_train@W - Y_train) + 2*reg*W)
		## END TODO

	return W, train_mses, test_mses

def weighted_regression(X, Y, r):
	'''
	Fill up this function for problem 3.
	Use closed form expression.
	r_train is a (n,) array, where n is number of training samples
	'''

	## TODO
	R = np.diag(r*r)

	W = (np.linalg.inv(X.T @ R @ X)) @ (X.T @ R @ Y)
	## END TODO
	return W

if __name__ == '__main__':
	# Load and split data
	X, Y = load_data2('train.csv')
	X, Y = preprocess(X, Y)
	X_train, Y_train, X_test, Y_test = split_data(X, Y)

	## TODO
	# Initialize W using using random normal 
	W = np.random.randn(X.shape[1], 1)
	## END TODO

	W, train_mses, test_mses = ordinary_least_squares(W, X_train, Y_train, X_test, Y_test)
	# W_ridge, train_mses, test_mses = ridge_regression(W, X_train, Y_train, X_test, Y_test, 10)

	# Plots
	plt.figure(figsize=(4,4))
	plt.plot(train_mses)
	plt.plot(test_mses)
	plt.legend(['Train MSE', 'Test MSE'])
	plt.xlabel('Iteration')
	plt.ylabel('MSE')
	plt.show()

