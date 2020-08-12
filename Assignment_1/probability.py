import numpy as np 
import argparse

def modify_uniform(u,lamda=1):
	'''
	This function modifies the given number assumed to be from a uniform distribution 

	TASK - Complete this function
	'''
	pass
	# return u



######################################


def generate_expo(n,lamda=1):
	'''
	Generates n IID samples from an exponential distribution
	'''
	exp = []
	for i in range(n):
		u = np.random.uniform(0,1)       # This generates a single number from a [0,1] uniform distribution
		x = modify_uniform(u,lamda=1)
		exp.append(x)
	return np.array(exp)   # This can be used to convert a list to a numpy array.

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Exponential Distribution')
	parser.add_argument('--num', type=int, default=100,
					help='Number of samples to generate')
	parser.add_argument('--seed', type=int, default=42,
					help='Seed for random generator')
	parser.add_argument('--lamda', type=float, default=1,
					help='Lambda parameter for the distribution')

	args = parser.parse_args()

	np.random.seed(args.seed)

	n = args.num

	exp = generate_expo(n,args.lamda)
	np.savetxt('problem_1.txt',exp,fmt='%1.4f')
	