import numpy as np
import utils, p1

def grade1():
    print("="*20 + "Grading Problem 1" + "="*20)
    marks = 0.0
    try:
        X, Y = utils.load_data2('data2.csv')
        X, Y = utils.preprocess(X, Y)
        X_train, Y_train, X_test, Y_test = utils.split_data(X, Y)
        W, train_mses, test_mses = p1.ista(X_train, Y_train, X_test, Y_test)
        assert train_mses[-1] < 0.2

        marks += 1.5
    except:
        print('Train Error is large')

    try:
        assert test_mses[-1] < 0.25
        marks += 1.5
    except:
        print('Test Error is large')
    print("Marks obtained in Problem 1: ", marks)
    return marks

print(f'Total Marks = {grade1()}')