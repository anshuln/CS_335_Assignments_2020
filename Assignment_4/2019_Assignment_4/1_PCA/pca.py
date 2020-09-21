import sys
import math
from numpy import load
from numpy.linalg import eig

def pca_small(X, k):
	## YOUR CODE BELOW

def pca_large(X, k):
	## YOUR CODE BELOW

def norm(v):
	""" Computes the L2 norm of a given vector. """
	return math.sqrt((v**2).sum())

def check_close(v1, v2, eps):
	""" Checks if v2 is equal to (+/-)v1. """
	return norm(v1-v2) < eps or norm(v1+v2) < eps

if __name__ == '__main__':
	if sys.argv[1] == 'small':
		X = load('X_small.npy')
		X_res = load('X_small_res.npy')
		X_stud = pca_small(X, X_res.shape[0])

		assert len(X_stud) == X_res.shape[0], \
			"number of components returned don't match"
		for i in range(X_res.shape[0]):
			assert check_close(X_res[i].squeeze(), X_stud[i].squeeze(), 1e-3), \
				"component %d is incorrect" % (i+1)

		print("PCA Small : Passed")

	elif sys.argv[1] == 'large':
		X = load('X_large_n.npy')
		X_res = load('X_large_n_res.npy')
		X_stud = pca_large(X, X_res.shape[0])

		assert len(X_stud) == X_res.shape[0], \
			"number of components returned don't match"
		for i in range(X_res.shape[0]):
			assert check_close(X_res[i].squeeze(), X_stud[i].squeeze(), 1e-3), \
				"file X_large_n.npy, component %d is incorrect" % (i+1)

		X = load('X_large_d.npy')
		X_res = load('X_large_d_res.npy')
		X_stud = pca_large(X, X_res.shape[0])

		assert len(X_stud) == X_res.shape[0], \
			"number of components returned don't match"
		for i in range(X_res.shape[0]):
			assert check_close(X_res[i].squeeze(), X_stud[i].squeeze(), 1e-3), \
				"file X_large_d.npy, component %d is incorrect" % (i+1)

		print("PCA Large : Passed")