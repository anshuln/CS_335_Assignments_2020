import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 1
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	pass

def logistic_train(X, Y, lr=0.01, max_iter = 500):
	''' TASK 1
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	lr 			= learning rate
	max_iter 	= maximum number of iterations of gradient descent to run
	Return the trained weight vector [D X 1] after performing gradient descent
	'''
	pass

def logistic_predict(X, Weights):
	''' TASK 1
	X 			= input feature matrix [N X D]
	Weights		= weight vector
	Return the predictions as [N X 1] vector
	'''
	pass
