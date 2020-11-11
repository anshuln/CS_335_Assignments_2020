import numpy as np
from kernel import *
from utils import *
import matplotlib.pyplot as plt

def gen_data(X_min=-30,X_max=30,y_min=-30,y_max=30,num_points=5000):
    X_coord = np.random.uniform(X_min,X_max,num_points)
    Y_coord = np.random.uniform(y_min,y_max,num_points)
    return np.column_stack([X_coord,Y_coord])

class KernelLogistic(object):
    def __init__(self, kernel=gaussian_kernel, iterations=100,eta=0.01,lamda=0.05,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.iterations = iterations
        self.alpha = None
        self.eta = eta     # Step size for gradient descent
        self.lamda = lamda # Regularization term

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.train_X = X
        self.train_y = y
        self.alpha = np.zeros((y.shape[0],1))
        kernel = self.kernel(self.train_X,self.train_X)

        # TODO
        for i in range(self.iterations):
            p1 = kernel@((self.lamda*self.alpha-y.reshape((y.shape[0],1))))
            exp = (np.exp(kernel@self.alpha))
            p2 = kernel@((exp/(1+exp)))
            self.alpha -= self.eta*(p1+p2) 
        # END TODO
    

    def predict(self, X):
        # TODO 
        exp = np.zeros((X.shape[0],))
        kernel = self.kernel(X,self.train_X)
        for j in range(X.shape[0]):
            exp[j] = (np.exp(-kernel[j,:]@self.alpha))
        return 1/(1+exp)
        # END TODO

def k_fold_cv(X,y,k=10,sigma=1.0):
    '''Does k-fold cross validation given train set (X, y)
    Divide train set into k subsets, and train on (k-1) while testing on 1. 
    Do this process k times.
    Do Not randomize 
    Arguments:
        X  -- Train set
        y  -- Train set labels
    
    Keyword Arguments:
        k {number} -- k for the evaluation
        sigma {number} -- parameter for gaussian kernel
    
    Returns:
        error -- (sum of total mistakes for each fold)/(k)
    '''
    # TODO 
    batch_size = len(X)//k
    best_model = None
    max_acc = -1
    errs = 0
    for i in range(k):
        X_train = np.row_stack((X[:i*batch_size,:],X[(i+1)*batch_size:,:]))
        y_train = np.array(y[:i*batch_size].tolist()+y[(i+1)*batch_size:].tolist())
        X_test = X[i*batch_size:(i+1)*batch_size,:]
        y_test = y[i*batch_size:(i+1)*batch_size]
        model = KernelLogistic(gaussian_kernel,100,0.01,0.05,sigma=sigma)
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test) > 0.5
        correct = np.sum(y_predict == y_test)
        errs += batch_size-correct
    return errs
    # END TODO

if __name__ == '__main__':
    data = np.loadtxt("./data/dataset1.txt")
    X1 = data[:900,:2]
    Y1 = data[:900,2]

    clf = KernelLogistic(gaussian_kernel)
    clf.fit(X1, Y1)

    y_predict = clf.predict(data[900:,:2]) > 0.5

    correct = np.sum(y_predict == data[900:,2])
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    if correct > 92:
        marks = 1.0
    else:
        marks = 0
    print(f"You recieve {marks} for the fit function")
    errs = []
    sigmas = [0.5, 1, 2, 3, 4, 5, 6]
    for s in sigmas:  
      errs+=[(k_fold_cv(X1,Y1,sigma=s))]
    plt.plot(sigmas,errs)
    plt.xlabel('Sigma')
    plt.ylabel('Mistakes')
    plt.title('A plot of sigma v/s mistakes')
    plt.show()
