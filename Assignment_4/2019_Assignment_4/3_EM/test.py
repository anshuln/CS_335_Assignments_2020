import os,sys
import numpy as np
import matplotlib.pyplot as plt
from GMM import *
np.random.seed(4)

num = 200
mean = [10,10]
cov = [[1,0],[0,1]] 
x1,y1 = np.random.multivariate_normal(mean,cov,num).T
plt.plot(x1,y1,'x',c='y')

mean = [2,10]
cov = [[0.5,0],[0,3]] 
x2,y2 = np.random.multivariate_normal(mean,cov,num).T
plt.plot(x2,y2,'x',c='g')

mean = [6,1]
cov = [[0.2,0],[0,2]]  
x3,y3 = np.random.multivariate_normal(mean,cov,num).T
plt.plot(x3,y3,'x',c='b')

X = np.concatenate((x1,x2,x3)).reshape(-1,1)
Y = np.concatenate((y1,y2,y3)).reshape(-1,1)
data = np.hstack((X, Y))
fit_gmm(data,3,30)
