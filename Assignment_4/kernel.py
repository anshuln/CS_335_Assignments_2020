import numpy as np 

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel
	
	[description]
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO 
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	return X @ Y.T
	pass
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf
	
	[description]
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - The sigma value for kernel
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	K = np.exp(-1.*((X[:,None,:] - Y[None,:,:])**2).sum(axis=-1)/(2 * sigma**2))
	return K
	# END TODO

def my_kernel(X,Y,sigma):
	'''[summary]
	
	[description]
	
	Arguments:
		X {[type]} -- [description]
		sigma- dummy argment, don't use
	''' 
	return X @ Y.T
	pass