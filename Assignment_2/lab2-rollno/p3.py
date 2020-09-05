import numpy as np 
from matplotlib import pyplot as plt
import argparse

from utils import *
from p1 import mse
def prepare_data(X,degree):
	'''
	X is a numpy matrix of size (n x 1)
	return a numpy matrix of size (n x (degree+1)), which contains higher order terms
	'''
	# TODO

	# End TODO
	return X 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Problem 4')
	parser.add_argument('--degree', type=int, default=3,
					help='Degree of polynomial to use')
	args = parser.parse_args()
	
	degree = args.degree

	X_train, Y_train = load_data1('data3_train.csv')
	X_test, Y_test   = load_data1('data3_test.csv')
	X_train = prepare_data(X_train,degree)
	X_test = prepare_data(X_test,degree)
	W = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ Y_train)

	train_mse = mse(X_train,Y_train,W)
	test_mse  = mse(X_test, Y_test, W)

	X_lin = np.linspace(X_train[:,1].min(),X_train[:,1].max()).reshape((50,1))
	X_lin = prepare_data(X_lin,degree)


	print(f'Train Error: %.4f Test Error: %4f '%(train_mse,test_mse))
	plt.scatter(X_train[:,1],Y_train)
	plt.scatter(X_test[:,1],Y_test)
	plt.plot(X_lin[:,1],X_lin @ W, c='g')
	plt.show()