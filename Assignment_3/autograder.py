import numpy as np
import perceptron
from binary_logistic_regression import *
from utils import *

def grade1():
	print("="*20 + "Grading Problem 1" + "="*20)
	marks = 0
	accs = [0.90, 0.85, 0.70, 0.50]
	try:
		X_train, Y_train, X_test, Y_test = perceptron.get_data('D2')

		assert perceptron.get_features(X_train[0]).size <=5, 'Atmost 5 features are allowed'
		
		X_train = np.array([perceptron.get_features(x) for x in X_train])
		X_test = np.array([perceptron.get_features(x) for x in X_test])

		C = max(np.max(Y_train), np.max(Y_test))+1
		D = X_train.shape[1]

		p = perceptron.Perceptron(C, D)

		p.train(X_train, Y_train)
		acc = p.eval(X_test, Y_test)

		if acc>=accs[0]:
			marks += 2.0
		elif acc>=accs[1]:
			marks += 1.5
		elif acc>=accs[2]:
			marks += 1.0
		elif acc>=accs[3]:
			marks += 0.5
	except:
		print('Error')
	print("Marks obtained in Problem 1: ", marks)
	return marks


def grade2():
	print("=" * 20 + "Grading Problem 2" + "=" * 20)
	marks = 0
	accs = [0.90, 0.85, 0.70, 0.50]
	try:
		X, Y = load_data('data/songs.csv')
		X, Y = preprocess(X, Y)
		X_train, Y_train, X_test, Y_test = split_data(X, Y)

		D = X_train.shape[1]

		lr = BinaryLogisticRegression(D)
		lr.train(X_train, Y_train)
		acc = lr.accuracy(X_test, Y_test)
		f1 = lr.f1_score(X_test, Y_test)
		print("=" * 20 + "Grading Problem 2.2(a)" + "=" * 20)
		if acc >= 0.8:
			marks += 3.0
		else:
			print("Test accuracy is too large.")

		print("=" * 20 + "Grading Problem 2.2(c)" + "=" * 20)
		try:
			assert f1 >= 0.25
			marks += 2.0
		except:
			print("Error in problem 2.2(c)")
	except:
		print('Error')
	print("Marks obtained in Problem 2: ", marks)
	return marks


print(f'Total Marks = {grade1() + grade2()}')