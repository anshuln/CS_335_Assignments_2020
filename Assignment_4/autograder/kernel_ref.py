import numpy as np 

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO 
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	return X @ Y.T
	# pass
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf
	
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
	'''Returns the gram matrix for your designed kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	return (1 + X @ Y.T)**4
	# pass
	# END TODO


def test():
	data = np.loadtxt("data/test_kernel_func.csv",delimiter=",")
	X = data[:,:2]
	Y = data[:,2:4]
	Z = gaussian_kernel(X,Y,sigma=0.1)
	W = linear_kernel(X,Y)
	assert np.allclose(W,data[:,4:14])
	assert np.allclose(Z,data[:,14:])


	# data = np.concatenate([X,Y,W,Z],axis=1)
	# np.savetxt("data/test_kernel_func.txt",data,delimiter=",")
# 
# test()
# if __name__ == "__main__":
