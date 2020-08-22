import numpy as np 
from utils import load_data2, split_data, preprocess, normalize
from multi_var_reg import weighted_regression
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Problem 4')
	parser.add_argument('--data', type=str, default='data4.csv',
					help='Path to csv file of dataset')
	args = parser.parse_args()

	X, Y = load_data2(args.data)
	X = X.astype('f')

	assert X.shape[1] >= 3

	W = weighted_regression(X,Y,np.ones((X.shape[0],)))

	print("Task 4 Complete")