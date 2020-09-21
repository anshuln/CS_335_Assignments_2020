import sys
import cluster
import itertools
from math import sqrt


ALLOWED_IMPORTS = [
    'from copy import deepcopy',
    'from itertools import cycle',
    'from pprint import pprint as pprint',
    'from array import array',
    'from cluster import distance_euclidean as distance',
    'import argparse',
    'import sys',
    'import argparse',
    'import matplotlib.pyplot as plt',
    'import random',
    'import math'
]


def compare(a, b):
    if isinstance(a, list):
        same = True
        for i,j in itertools.zip_longest(a,b):
            try:
                same = same and isclose(i,j)
            except Exception:
                same = same and (i==j)
    else:
        try:
            same = isclose(a,b)
        except Exception:
            same = (a==b)
    return same


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def check_imports(filename):
    with open(filename,'r') as f:
        data = f.readlines()

    imports = [i.strip() for i in data if i.find('import') >= 0]
    not_allowed = set(imports) - set(ALLOWED_IMPORTS)
    if len(not_allowed) > 0:
        print('You are not allowed to import anything already present in the code. Please remove the following from {}:'.format(filename))
        for i in not_allowed:
            print(i)
        sys.exit(0)


def grade1():
    print('='*20 + ' TASK 1 ' + '='*20)
    testcases = {
        'distance_euclidean': [
            (((2,2), (2,2)), 0),
            (((0,0), (0,1)), 1),
            (((1,0), (0,1)), sqrt(2)),
            (((0,0), (0,-1)), 1),
            (((0,0.5), (0,-0.5)), 1)
        ],
        'kmeans_iteration_one': [
            (([(i,i) for i in range(5)], [(1,1),(2,2),(3,3)], cluster.distance_euclidean), [(0.5, 0.5), (2.0, 2.0), (3.5, 3.5)]),
            (([(i+1,i*2.3) for i in range(5)], [(5,1),(-1,2),(3,6)], cluster.distance_euclidean), [(5, 1), (1.5, 1.15), (4.0, 6.8999999999999995)])
        ],
        'hasconverged': [
            (([(i,i*2,i*3) for i in range(5)], [(i,i*2,i*3+0.01) for i in range(5)], 0.01), True),
            (([(i,i*2,i*3) for i in range(5)], [(i,i*2,i*3+0.01) for i in range(5)], 0.002), False)
        ],
        'iteration_many': [
            (([(i,i) for i in range(3)], [(1,1),(2,2)], cluster.distance_euclidean, 3, 'kmeans', 0.01), [[(1, 1), (2, 2)], [(0.5, 0.5), (2.0, 2.0)], [(0.5, 0.5), (2.0, 2.0)]]),
            (([(i+1,i*2.3) for i in range(3)], [(5,1),(-1,2)], cluster.distance_euclidean, 5, 'kmeans', 0.01), [[(5, 1), (-1, 2)], [(3.0, 4.6), (1.5, 1.15)], [(3.0, 4.6), (1.5, 1.15)]])
        ],
        'performance_SSE': [
            (([(0,i) for i in range(10)], [(0,0), (0,5), (0,10)], cluster.distance_euclidean), 20),
            (([(0,i) for i in range(10)], [(0,0), (0,5), (0,6), (0,10)], cluster.distance_euclidean), 16),
            (([(0,i) for i in range(10)], [(0,50), (-2,5.8), (3,6.1), (0.5,10)], cluster.distance_euclidean), 121.82)
        ]
    }
    grade = 0
    for function in testcases:
        passed = True
        for inp, out in testcases[function]:
            try:
                ret = getattr(cluster,function)(*inp)
            except Exception:
                ret = None
            if not compare(ret, out):
                print('Function {} failed a testcase.\n\tInput: {}\n\tExpected return: {}\n\tRecieved return: {}\n'.format(function, str(inp)[1:-1], out, ret))
                passed = False
        print('  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), function))
        print('-'*30)
        grade += passed

    passed = 1
    for n,k in [(3,3),(10,3),(20,15)]:
        data = [(i,i) for i in range(n)]
        try:
            ret = cluster.initialization_forgy(data, k)
        except Exception:
            ret = None
        if ret is None or len(ret) != k:
            passed = 0
        else:
            passed = sum([1 for i in ret if i not in data]) == 0
        if passed == 0:
            print('Function initialization_forgy failed a testcase.\n\tInput: {}\n\tExpected return: All cluster centers must come from data points.\n\tRecieved return: {}\n'.format((data, k), ret))
    print('  {}  Function {}'.format([u'\u2718', u'\u2713'][passed].encode('utf8'), 'initialization_forgy'))
    print('-'*30)


    grade = (grade+passed)*0.5

    print('grade: {}'.format(grade))
    print('')
    return grade


def grade2():
    print('='*20 + ' TASK 2 ' + '='*20)
    print('This task is manually graded. Answer it in the file solutions.txt\n')
    return 0



def gradeall(loc):
    print('='*48 + '\nFINAL GRADE: {}\n\n'.format(sum([loc['grade' + str(i)]() for i in range(1,3)])))


for filename in ['cluster.py']:
    check_imports(filename)

if len(sys.argv) < 2:
    print('usage:\npython autograder.py [task-number]\npython autograder.py all')
    sys.exit(1)
print('')
if sys.argv[1].lower() == 'all':
    gradeall(locals())
else:
    locals()['grade' + str(int(sys.argv[1]))]()
