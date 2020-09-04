import numpy as np

def get_data(dataset):
    datasets = ['D1', 'D2']
    assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
    X_train = np.loadtxt(f'data/{dataset}/training_data')
    Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
    X_test = np.loadtxt(f'data/{dataset}/test_data')
    Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

    return X_train, Y_train, X_test, Y_test

class Perceptron():
    def __init__(self, C, D):
        '''
        C - number of classes
        D - number of features
        '''
        self.C = C
        self.weights = np.zeros((C, D))
        
    def pred(self, x):
        return np.argmax(self.weights@(x.reshape(-1, 1)))

    def train(self, X, Y, max_iter):
        for _ in range(max_iter):
            for i in range(X.shape[0]):
                ### TODO
                pred = self.pred(X[i])
                if pred != Y[i]:
                    self.weights[Y[i]] += X[i]
                    self.weights[pred] -= X[i]
                ### END TODO

    def eval(self, X, Y):
        n_samples = X.shape[0]
        correct = 0
        for i in range(X.shape[0]):
            if self.pred(X[i]) == Y[i]:
                correct += 1
        print(f'Accuracy: {correct/n_samples}')

X_train, Y_train, X_test, Y_test = get_data('D2')

C = max(np.max(Y_train), np.max(Y_test))+1
D = X_train.shape[1]

perceptron = Perceptron(C, D)

perceptron.train(X_train, Y_train, 10)
perceptron.eval(X_test, Y_test)
