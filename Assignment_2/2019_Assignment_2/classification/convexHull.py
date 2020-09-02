import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
random.seed(42)


def visualize(points):
	''' Write the code here '''
	

def grade():
	A,B = [],[]
	for i in range(3):
		A.append('A'+str(i))
		B.append('B'+str(i))

	points  = [3,5,7]
	allpoints = []
	till = 3
	for i in range(till):
		coords = np.array([(A[i], random.random()*(100.0/points[i]), random.random()*(100.0)) for _ in range(points[i])])
		coords1 = np.array([(B[i], 25 + random.random()*(100.0/points[i]), random.random()*(100.0)) for _ in range(points[i])]) 
		allpoints.extend(coords)
		allpoints.extend(coords1)

	random.shuffle(allpoints)
	visualize(allpoints)
	return allpoints

