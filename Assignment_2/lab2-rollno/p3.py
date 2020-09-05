import numpy as np 
from matplotlib import pyplot as plt
import argparse

from utils import *
from p1 import mse

## ONLY CHANGE CODE BETWEEN TODO and END TODO
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
	np.random.seed(42)
	degree = args.degree

	X_train, Y_train = load_data1('data3_train.csv')
	Y_train = Y_train/20
	X_test, Y_test   = load_data1('data3_test.csv')
	Y_test = Y_test/20

	X_train = prepare_data(X_train,degree)
	indices_0 = np.random.choice(np.arange(200),40,replace=False)
	indices_1 = np.random.choice(np.arange(200),40,replace=False)
	indices_2 = np.random.choice(np.arange(200),40,replace=False)
	indices_3 = np.random.choice(np.arange(200),40,replace=False)

	## TODO - compute each fold using indices above, compute weights using OLS
	X_0 = 
	Y_0 = 
	X_1 = 
	Y_1 = 
	X_2 = 
	Y_2 = 
	X_3 = 
	Y_3 = 
	W_0 = 
	W_1 = 
	W_2 = 
	W_3 = 

	## END TODO


	X_test = prepare_data(X_test,degree)

	train_mse_0 = mse(X_0,Y_0,W_0)
	train_mse_1 = mse(X_1,Y_1,W_1)
	train_mse_2 = mse(X_2,Y_2,W_2)
	train_mse_3 = mse(X_3,Y_3,W_3)
	test_mse_0  = mse(X_test, Y_test, W_0)
	test_mse_1  = mse(X_test, Y_test, W_1)
	test_mse_2  = mse(X_test, Y_test, W_2)
	test_mse_3  = mse(X_test, Y_test, W_3)

	X_lin = np.linspace(X_train[:,1].min(),X_train[:,1].max()).reshape((50,1))
	X_lin = prepare_data(X_lin,degree)

	print(f'train Error 1: %.4f train Error 2: %4f train Error 3: %4f train E 4: %.4f'%(train_mse_0,train_mse_1,train_mse_2,train_mse_3))
	print(f'Test Error 1: %.4f Test Error 2: %4f Test Error 3: %4f test E 4: %.4f'%(test_mse_0,test_mse_1,test_mse_2,test_mse_3))
	plt.scatter(X_train[:,1],Y_train,color='orange')
	plt.plot(X_lin[:,1],X_lin @ W_0, c='g')
	plt.plot(X_lin[:,1],X_lin @ W_1, c='r')
	plt.plot(X_lin[:,1],X_lin @ W_2, c='b')
	plt.plot(X_lin[:,1],X_lin @ W_3, color='purple')
	plt.show()