import numpy as np 

X = np.random.uniform(0,4*3.1415,10)
arr = np.vstack([X/(4*3.1415),np.sin(X)+1.5*np.cos(X) + 0.5*np.sin(2*X) -0.75*np.cos(2*X) +
					1.75*np.sin(3*X)-2.5*np.cos(3*X) - 1.5*np.sin(5*X) + 1.25*np.cos(5*X) +
					1.2*np.sin(7*X)-2.15*np.cos(9*X) + 1.35*np.sin(11*X) - 0.5*np.cos(17*X)+
					1.5*np.sin(19*X)-2.15*np.cos(4*X) + 1.35*np.sin(8*X) - 0.5*np.cos(6*X)]).T
np.savetxt("data3_train_1.csv",arr, delimiter=",")

X = np.random.uniform(0,4*3.1415,10)
arr = np.vstack([X/(4*3.1415),np.sin(X)+1.5*np.cos(X) + 0.5*np.sin(2*X) -0.75*np.cos(2*X) +
					1.75*np.sin(3*X)-2.5*np.cos(3*X) - 1.5*np.sin(5*X) + 1.25*np.cos(5*X) +
					1.2*np.sin(7*X)-2.15*np.cos(9*X) + 1.35*np.sin(11*X) - 0.5*np.cos(17*X)+
					1.5*np.sin(19*X)-2.15*np.cos(4*X) + 1.35*np.sin(8*X) - 0.5*np.cos(6*X)]).T
np.savetxt("data3_test.csv",arr, delimiter=",")