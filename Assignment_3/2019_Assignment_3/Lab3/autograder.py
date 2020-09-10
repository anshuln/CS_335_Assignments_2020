import sys, math
import numpy as np
from task import *
from utils import *
import matplotlib.pyplot as plt
from kernel_perceptron import *

def grade2():
    print('='*20 + ' TASK 2 - Kernel Perceptron' + '='*20)
    try:
        num,t = 1000,1600
        X1, y1, X2, y2 = gen_non_lin_separable_data(num)
        X_train, y_train = split_train(X1, y1, X2, y2, t)
        X_test, y_test = split_test(X1, y1, X2, y2, t)

        clf = KernelPerceptron(gaussian_kernel, iterations=20)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        accuracy = correct/len(y_predict)

        if accuracy>0.85:
            marks = 1.5
            print("TASK 2 Passed")
        else:
            marks = 0
            print("Your Accuracy : ", accuracy)
            print("Expected Accuracy : ", 0.85)
            print("TASK 2 Failed")

    except KeyError as e:    
        marks = 0
        print("RunTimeError in TASK 2 : " + str(e))
        print("TASK 2 Failed")    

    return marks


def grade1(trainX, trainY, testX, testY):
	print('='*20 + ' TASK 1 - Logistic Regression' + '='*20)
	try:
		weights = logistic_train(trainX,trainY)
		print(np.shape(weights))
		preds = logistic_predict(testX,weights)
		accuracy = ((preds == np.around(testY)).sum())/len(preds)
		if accuracy>0.885:
			marks = 1.5
			print("Your Accuracy : ",accuracy)
			print("TASK 1 Passed")
		else:
			marks = 0
			print("Your Accuracy : ", accuracy)
			print("Expected Accuracy : >", 0.885)
			print("TASK 1 Failed")

	except KeyError as e:    
		marks = 0
		print("RunTimeError in TASK 1 : " + str(e))
		print("TASK 1 Failed")     
	
	return marks

if __name__ == "__main__":
    X1, Y1 = read_data("./dataset/bank.csv")
    X1, Y1 = preprocess(X1,Y1)
    xtrain,ytrain,xtest,ytest = separate_data_randomize(X1,Y1)

    if len(sys.argv) < 2:
    	print('usage:\npython autograder.py [task-number]\npython autograder.py all')
    	sys.exit(1)

    if sys.argv[1].lower() == 'all':
        m2 = grade2()
        m1 = grade1(xtrain,ytrain,xtest,ytest)
        print('='*48 + '\nFINAL GRADE: {}/3\n\n'.format(m2 + m1))
    elif int(sys.argv[1]) == 2:
        m2 = grade2()
    elif int(sys.argv[1]) == 1:
        m1 = grade1(xtrain,ytrain,xtest,ytest)