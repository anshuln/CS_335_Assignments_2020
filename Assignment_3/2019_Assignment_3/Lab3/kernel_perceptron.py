import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    pass

def polynomial_kernel(x, y, degree=3):
    pass

def gaussian_kernel(x, y, sigma=5.0):
    pass

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, iterations=1):
        self.kernel = kernel
        self.iterations = iterations
        self.alpha = None

    def fit(self, X, y):
        ''' find the alpha values here'''
        pass
    
    def project(self, X):
        '''return projected values from alpha and corresponding support vectors'''
        pass

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))
