import numpy as np
import p2

def gradex():
	print("="*20 + "Grading Problem ???" + "="*20)
	marks = 0
	accs = [0.85, 0.70, 0.50]
	try:
		X_train, Y_train, X_test, Y_test = p2.get_data('D2')

		assert p2.get_features(X_train[0]).size <=5, 'Atmost 5 features are allowed'
		
		X_train = np.array([p2.get_features(x) for x in X_train])
		X_test = np.array([p2.get_features(x) for x in X_test])

		C = max(np.max(Y_train), np.max(Y_test))+1
		D = X_train.shape[1]

		p = p2.Perceptron(C, D)

		p.train(X_train, Y_train)
		acc = p.eval(X_test, Y_test)

		if acc>=accs[0]:
			marks += 2.0
		elif acc>=accs[1]:
			marks += 1
		elif acc>=accs[2]:
			marks += 0.5
	except:
		print('Error')
	print("Marks obtained in Problem ???: ", marks)
	return marks


print(f'Total Marks = {gradex()}')