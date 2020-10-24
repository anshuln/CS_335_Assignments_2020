import sys
import test
import numpy as np
from layers import *
import pickle

def grade1():
    pathToTestCase = "testcases/testcase_01.pkl"
    load_test_case = pickle.load(open(pathToTestCase, 'rb'))

    def check_relu(task_number=1.1):
        print('='*20 + ' TASK '+str(task_number)+ '='*20)
        input_X = np.asarray(load_test_case['relu_input']).reshape(1,4)


        output_X = input_X

        output_X = relu_of_X(output_X)

        studentAnswer = output_X
        teacherAnswer = load_test_case['relu_output']

        teacherAnswer = np.round(teacherAnswer, 6)
        studentAnswer = np.round(studentAnswer, 6)

        print('Student Answer', studentAnswer)
        print('Correct Answer', teacherAnswer)
        print('Correct', np.array_equal(studentAnswer, teacherAnswer))
        return np.array_equal(studentAnswer, teacherAnswer)

    def check_gradient_relu(task_number=1.2):
        print('='*20 + ' TASK '+str(task_number)+ '='*20)
        input_X = np.asarray(load_test_case['gardient_relu_input']).reshape(1,4)
        input_delta = np.asarray(load_test_case['gardient_relu_input']).reshape(1,4)

    
        output_X = input_X

        output_X = gradient_relu_of_X(output_X, input_delta)

        studentAnswer = output_X
        teacherAnswer = load_test_case['gardient_relu_output']

        teacherAnswer = np.round(teacherAnswer, 6)
        studentAnswer = np.round(studentAnswer, 6)

        print('Student Answer', studentAnswer)
        print('Correct Answer', teacherAnswer)
        print('Correct', np.array_equal(studentAnswer, teacherAnswer))
        return np.array_equal(studentAnswer, teacherAnswer)


    def check_softmax(task_number=1.3):
        print('='*20 + ' TASK '+str(task_number)+ '='*20)
        input_X = np.asarray(load_test_case['softmax_input']).reshape(1,4)

        output_X = input_X

        output_X = softmax_of_X(output_X)

        studentAnswer = output_X

        teacherAnswer = load_test_case['softmax_output']

        teacherAnswer = np.round(teacherAnswer, 6)
        studentAnswer = np.round(studentAnswer, 6)

        print('Student Answer', studentAnswer)
        print('Correct Answer', teacherAnswer)
        print('Correct', np.array_equal(studentAnswer, teacherAnswer))
        return np.array_equal(studentAnswer, teacherAnswer)

    def check_gradient_softmax(task_number=1.4):
        print('='*20 + ' TASK '+str(task_number)+ '='*20)
        input_X = np.asarray(load_test_case['gardient_softmax_input']).reshape(1,4)
        input_delta = np.asarray(load_test_case['gardient_softmax_input_delta']).reshape(1,4)

        output_X = input_X

        output_X = gradient_softmax_of_X(output_X, input_delta)
        
        studentAnswer = output_X
        teacherAnswer = load_test_case['gardient_softmax_output']

        teacherAnswer = np.round(teacherAnswer, 6)
        studentAnswer = np.round(studentAnswer, 6)

        print('Student Answer', studentAnswer)
        print('Correct Answer', teacherAnswer)
        print('Correct', np.array_equal(studentAnswer, teacherAnswer))
        return np.array_equal(studentAnswer, teacherAnswer)




    np.random.seed(42)
    print('='*20 + ' TASK 1 - Forward Pass' + '='*20)
    marks = 0

    try:
        if check_relu():
            marks+= 1
    except:
        print("Error in Test Case 1.1")

    try:
        if check_gradient_relu():
            marks+= 1
    except:
        print("Error in Test Case 1.2")

    try:
        if check_softmax():
            marks+= 1
    except:
        print("Error in Test Case 1.3")

    try:
        if check_gradient_softmax():
            marks+= 2
    except:
        print("Error in Test Case 1.1")

    
    print('Marks: {}/5'.format(marks))
    return marks

def grade2():
    print('='*20 + ' TASK 2 - Forward + Backward Pass' + '='*20)
    marks = 0
    
    try:
        net, xtest, ytest = test.task[1](False)
        marks += test_net(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.1")


    try:
        net, xtest, ytest = test.task[2]()
        marks += 3 * test_net(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.3")


    try:
        net, xtest, ytest, name = test.task[3]()
        model = np.load(name)
        k,i = 0,0
        for l in net.layers:
            if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
                net.layers[i].weights = model[k]
                net.layers[i].biases = model[k+1]
                k+=2
            i+=1

        marks += 4 * test_net_diff(net, xtest, ytest)
    except:
        print("RunTimeError in Task 2.4")
    

    print('Marks: {}/9'.format(marks))
    return marks


def test_net(net, xtest, ytest):
    _, acc  = net.validate(xtest, ytest)
    if acc >= 90:
        return 1
    elif acc >= 85:
        return 0.75
    elif acc >= 75:
        return 0.5
    else:
        return 0

def test_net_diff(net, xtest, ytest):
    _, acc  = net.validate(xtest, ytest)
    print(acc)
    if acc >= 35:
        return 1
    elif acc >= 30:
        return 0.75
    elif acc >= 25:
        return 0.5
    else:
        return 0

if len(sys.argv) < 3:
    print('usage:\npython3 autograder.py -t task-number')
    sys.exit(1)

locals()['grade' + str(int(sys.argv[2]))]()



