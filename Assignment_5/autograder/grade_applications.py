import io
from contextlib import redirect_stdout
import pickle

import numpy as np

import nn
from util import *
from layers import *
from trainer import Trainer

marks_map = {
                'XOR': [(90, 2), (85, 1)], 
                'CIFAR10': [(35, 4), (32, 3), (30, 2), (25, 1)],
            }

def test_applications(dataset, seeds=[]):
    '''
    seeds -> list of seeds to test on
    '''
    marks = 0

    np.random.seed(337)
    seeds_ = list(np.random.randint(1, 20000, 10))
    seeds = (seeds+seeds_)[:10]

    acc = 0
    for seed in seeds:
        trainer = Trainer(dataset)
        
        f = io.StringIO()
        with redirect_stdout(f):
            trainer.train(verbose=False)
        out = f.getvalue()
        acc += float(out.strip().split(' ')[-1])
        print(out.strip())

    acc = acc/10.0

    print(acc)

    for i, j in marks_map[dataset]:
        if acc > i:
            marks += j
            break

    return marks

def test_cifar(seed=335):
    '''
    seeds -> list of seeds to test on
    '''
    marks = 0

    np.random.seed(seed)

    trainer = Trainer("CIFAR10")

    try:
        model = pickle.load(open('model.p', 'rb'))
    except:
        try:
           model = pickle.load(open('model.npy', 'rb'))
        except:
            print("Saved model not found") 

    if 'ConvolutionLayer' not in [type(l).__name__ for l in trainer.nn.layers]:
        print('ConvolutionLayer not used')
        return 0

    i = 0
    for l in trainer.nn.layers:
        if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
            l.weights = model[i]
            l.biases = model[i+1]
            i = i + 2
    print("Model Loaded... ")

    _, acc = trainer.nn.validate(trainer.XTest, trainer.YTest)
    print(acc)
    
    for i, j in marks_map['CIFAR10']:
        if acc > i:
            marks += j
            break

    return marks

# print(test_applications('XOR'))
print(test_cifar())
