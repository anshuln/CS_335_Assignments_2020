'''Main script to run the code

Usage - python3 main.py --dataset [] --verbose

'''

import argparse
from trainer import * 
from util import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	""" Arguments: arg """
	parser.add_argument('--dataset')
	parser.add_argument('--verbose',action='store_true')
	
	args = parser.parse_args()
	trainer = Trainer(args.dataset)
	trainer.train(args.verbose)

