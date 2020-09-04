import numpy as np
import utils, p1, perceptron

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

def grade2():
    print("="*20 + "Grading Problem 2" + "="*20)
    marks = 0
    accs = [[0.78, 0.70, 0.60], [0.97, 0.90, 0.80]]
    try:
        for i, ds in enumerate(['D1', 'D2']):
            X_train, Y_train, X_test, Y_test = perceptron.get_data(ds)

            C = max(np.max(Y_train), np.max(Y_test))+1
            D = X_train.shape[1]

            p = perceptron.Perceptron(C, D)

            p.train(X_train, Y_train)
            acc = p.eval(X_test, Y_test)

            if acc>=accs[i][0]:
                marks += 1.5
            elif acc>=accs[i][1]:
                marks += 1
            elif acc>=accs[i][2]:
                marks += 0.5
    except:
        print('Error')
    print("Marks obtained in Problem 2: ", marks)
    return marks

print(f'Total Marks = {grade1() + grade2()}')