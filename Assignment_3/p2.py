import numpy as np
import argparse

def get_data(dataset,num_train_samples=-1):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train[:num_train_samples,:], Y_train[:num_train_samples], X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )

	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO
	image = x.reshape(50, 50).T

	s1 = 0.0
	s2 = 0.0
	s3 = 0.0
	for x in range(50): 
		for y in range(50):
			v1 = abs(image[(x-1)%50,y]-image[(x+1)%50,y])/2.0
			v2 = abs(image[x,(y-1)%50]-image[x,(y+1)%50])/2.0
			s1 += (v1**2+v2**2)**0.5
			s2 += image[x,y]
			s3 += v1

	return np.array([s1, s2/s1*107])
	### END TODO

class Perceptron():
	def __init__(self, C, D):
		'''
		C - number of classes
		D - number of features
		'''
		self.C = C
		self.weights = np.zeros((C, D))
		
	def pred(self, x):
		'''
		x - numpy array of shape (D,)
		'''
		### TODO: Return predicted class for x
		return np.argmax(self.weights@(x.reshape(-1, 1)))
		### END TODO

	def train(self, X, Y, max_iter=10):
		for iter in range(max_iter):
			for i in range(X.shape[0]):
				### TODO: Update weights
				pred = self.pred(X[i])
				if pred != Y[i]:
					self.weights[Y[i]] += X[i]
					self.weights[pred] -= X[i]
				### END TODO
			# print(f'Train Accuracy at iter {iter} = {self.eval(X, Y)}')

	def eval(self, X, Y):
		n_samples = X.shape[0]
		correct = 0
		for i in range(X.shape[0]):
			if self.pred(X[i]) == Y[i]:
				correct += 1
		return correct/n_samples

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Problem 4')
	parser.add_argument('--num_samples', type=int, default=-1,
					help='Number of samples to train on')
	args = parser.parse_args()
	
	num_train_samples = args.num_samples

	X_train, Y_train, X_test, Y_test = get_data('D2',num_train_samples)

	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')
