from copy import deepcopy
from itertools import cycle
from pprint import pprint as pprint
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
import math



def RBFKernel(p1,p2,sigma=3):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point
	Returns the value of RBF kernel
	'''
	value = None
	# TODO [task3]
	# Your function must work for all sized tuples.
	return value

def initializationrandom(data,C,seed=45):
	'''
	data: list of tuples: the list of data points
	C:int : number of cluster centroids
    seed:int : seed value for random value generator
	Returns a list of tuples,representing the cluster centroids and a list of list of tuples representing the cluster  
	'''
	centroidList =  []
	clusterLists = [[] for i in range(C)]
	# TODO [task3]:
	# Initialize the cluster centroids by sampling k unique datapoints from data and assigning a data point to a random cluster
	assert len(centroidList) == C
	assert len(clusterLists) == C
	return centroidList,clusterLists

def firstTerm(p):
	'''
	p: a tuple for a  datapoint
	'''
	value = None
	'''
	# TODO [task3]:
	# compute the first term in the summation of distance.
	'''
	return value

def secondTerm(data,pi_k):
	'''
	data : list of tuples: the list of data points
	pi_k : list of tuples: the list of data points in kth cluster
	'''
	value = None
	'''
	# TODO [task3]:
	# compute the second term in the summation of distance.
	'''
	return value

def thirdTerm(pi_k):
	'''
	pi_k : list of tuples: the list of data points in kth cluster
	'''
	value = None
	'''
	# TODO [task3]:
	# compute the third term in the summation of distance.
	'''
	return value

def hasconverged(prevclusterList,clusterList,C):
	'''
	prevclusterList : list of (list of tuples): the list of lists of  tuples of datapoints in a cluster in previous iteration
	clusterList: list of (list of tuples): the list of lists of tuples of datapoints in a cluster
	C: int : number of clusters
	'''
	converged = False
	'''
	# TODO [task3]:
	check if the cluster membership of the clusters has changed or not.If not,return True. 
	'''
	return converged 
	
def kernelkmeans(data,C,maxiter=10);
	'''
	data : list of tuples: the list of data points
	C: int : number of clusters
	'''
	centroidList,clusterLists = initializationrandom(data,C)
	'''
	# TODO [task3]:
	# iteratively modify the cluster centroids.
	# Stop only if convergence is reached, or if max iterations have been exhausted.
	# Save the results of each iteration in all_centroids.
	# Tip: use deepcopy() if you run into weirdness.
	'''
	return clusterLists,centroidList




def plot(clusterLists,centroidList,C):
	color = iter(cm.rainbow(np.linspace(0,1,C)))
    plt.figure("result")
    plt.clf()
    for i in range(C):
        col = next(color)
        memberCluster = np.asmatrix(listClusterMembers[i])
        plt.scatter(np.ravel(memberCluster[:,0]),np.ravel(memberCluster[:,1]),marker=".",s =100,c = col)
    color = iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(n):
        col = next(color)
        plt.scatter(np.ravel(centroid[i,0]),np.ravel(centroid[i,1]),marker="*",s=400,c=col,edgecolors="black")
        plt.show()


filePath1 = "datasets/mouse.csv"
filePath2 = "datasets/3lines.csv"
mouse  = np.loadtxt(open(filePath1, "rb"), delimiter=",", skiprows=1)
lines3 = np.loadtxt(open(filePath2, "rb"), delimiter=",", skiprows=1)
clusterResult, centroid = kernelkmeans(mouse,C=3)
plotResult(clusterResult, centroid,C)
clusterResult,centroid = kernelkmeans(lines3,C=3)
plotResult(clusterResult, centroid,C)
#save the plots accordingly

