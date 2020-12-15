import io
from contextlib import redirect_stdout

import numpy as np

import nn
from util import *
from layers import *
from trainer import Trainer

def test_applications(dataset, seeds=[]):
    '''
    seeds -> list of seeds to test on
    '''
    marks_map = {'XOR': [(90, 2), (85, 1)]}
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
        if acc >= i:
            marks += j
            break

    return marks

print(test_applications('XOR'))
