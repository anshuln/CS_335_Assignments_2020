import numpy as np
from kmeans import KMeans
from kernel import *
from kernel_ref import linear_kernel as lk_ref 
from kernel_ref import gaussian_kernel as gk_ref 
from krr_ref import KernelRidgeRegression as KRR_ref 
from krr import *
from kernel_logistic import *
from kernel_logistic_ref import KernelLogistic as KL_ref
import time

np.random.seed(37)
def grade4():
	print("==============================\n Grading KMeans")
	marks = 0
	try:
		data = np.array([[i,i] for i in range(5)])
		centers = np.array([[1.,1.], [2.,2.], [3.,3.]])
		op = np.array([[0.5, 0.5], [2.0, 2.0], [3.5, 3.5]])

		kmeans = KMeans(D=2, n_clusters=3)
		kmeans.cluster_centers = centers
		it = kmeans.train(data, 1)
		if np.allclose(kmeans.cluster_centers, op) and it==0:
			marks += 0.5

		data = np.array([[i+1,i*2.3] for i in range(5)])
		centers = np.array([[5.,1.], [-1.,2.], [3.,6.]])
		op = np.array([[5, 1], [1.5, 1.15], [4.0, 6.8999999999999995]])

		kmeans = KMeans(D=2, n_clusters=3)
		kmeans.cluster_centers = centers
		it = kmeans.train(data, 1)
		if np.allclose(kmeans.cluster_centers, op) and it==0:
			marks += 0.5

		data = np.array([[i+1,i*2.3] for i in range(3)])
		centers = np.array([[5, 1], [-1., 2]])
		op = np.array([[3.0, 4.6], [1.5, 1.15]])
		kmeans = KMeans(D=2, n_clusters=2)
		kmeans.cluster_centers = centers
		it = kmeans.train(data, 5)
		if np.allclose(kmeans.cluster_centers, op) and it==1:
			marks += 1
	except:
		print('Error in k-means')
	print(f"Marks obtained {marks}")
	return marks

def grade1():
	# data = np.loadtxt("data/test_kernel_func.csv",delimiter=",")
	# X = data[:,:2]
	# Y = data[:,2:4]
	print("==============================\n Grading Kernel Functions")
	X = np.random.normal(0,1,(1000,2))
	Y = np.random.normal(0,1,(1000,2))
	marks = 0
	try:
		t_student = time.time()
		Z = gaussian_kernel(X,Y,sigma=0.1)
		t_student = time.time() - t_student 

		t_ref = time.time()
		Z_ref = gk_ref(X,Y,sigma=0.1)
		t_ref = time.time() - t_ref

		if np.allclose(Z_ref,Z,rtol=0): 
			marks += 0.5

			if t_student < 3*t_ref:
				marks += 1
			else:
				print("RBF computation is too slow")
		else:
			print("Wrong output for RBF kernel")

		W = linear_kernel(X,Y)
		W_ref = lk_ref(X,Y)
		if np.allclose(W,W_ref):
			marks += 0.5
		else:
			print("Wrong output for linear_kernel")
	except Exception as e:
		print("Error in kernel functions")
		print(e)

	print(f"Marks obtained {marks}")

	return marks
def grade2():
	print("==============================\n Grading Kernel Logistic Regression")
	marks = 0
	data = np.loadtxt("./data/dataset1.txt")
	X1 = data[:900,:2]
	Y1 = data[:900,2]
	try:
		clf = KernelLogistic(gaussian_kernel)
		clf.fit(X1, Y1)

		y_predict = clf.predict(data[900:,:2]) > 0.5
	except Exception as e:
		print("Error in your kernel_logistic")
		print(e)
		marks = 0
		return marks
	correct = np.sum(y_predict == data[900:,2])
	if correct > 92:
		marks += 1.0
	else:
		marks += 0

	clf_ref = KL_ref(gaussian_kernel)
	clf_ref.train_X = X1
	clf_ref.alpha = clf.alpha

	y_ref = clf_ref.predict(data[900:,:2])
	y_     = clf.predict(data[900:,:2])
	if np.allclose(y_,y_ref,rtol=1e-8, atol=1e-8):
		marks += 0.5
	else:
		print("Predict function for kernel_logistic is incorrect")


	# Todo - better check for k_fold?
	try:

		errs = []
		sigmas = [0.5, 1, 2, 3, 4, 5, 6]
		for s in sigmas:  
			errs+=[(k_fold_cv(X1,Y1,sigma=s))]

		if errs[1] < errs[0] and errs[1] < errs[2]:
			marks += 1.5
		else:
			print("Error in k_fold_cv")
	except Exception as e:
		print("Implementation error in your k_fold_cv")
		print(e)
	print(f"Marks obtained {marks}")

	return marks
	


def grade3():
	# TODO - should we also check efficiency?
	print("==============================\n Grading Kernel Ridge Regression")
	marks = 0
	X, Y = read_data('./data/krr.csv')
	try:
		clf = KernelRidgeRegression(gaussian_kernel,0.01,10)
		clf.fit(X, Y)
	except Exception as e:
		print("Implementation error in krr")
		print(e)
		return marks
	clf_ref = KRR_ref(gaussian_kernel,0.01,10)
	clf_ref.fit(X,Y)
	if np.allclose(clf_ref.alpha,clf.alpha,rtol=1e-8):
		marks += 1
	else:
		print("Fit function for krr is incorrect")

	clf_ref.alpha = clf.alpha

	try:
		Y_    = clf.predict(X)
	except Exception as e:
		print("Implementation error in krr")
		print(e)
		return marks

	Y_ref = clf_ref.predict(X)

	if np.allclose(Y_,Y_ref,rtol=1e-8):
		marks += 0.5
	else:
		print("Predict function for krr is incorrect")
	print(f"Marks obtained {marks}")

	return marks

def grade5():
	print("==============================\n Grading Kernel Design")
	marks = 0
	try:
		X, Y = read_data('./data/kernel_design.csv')
		# plot_3D(X[:,0], X[:,1], Y)
		clf = KernelRidgeRegression(my_kernel,0.01,0.1)
		clf.fit(X, Y)

		err = np.linalg.norm(clf.predict(X)-Y)**2
		# plt.show()

		if err < 7000:
			marks += 1.0
		elif err < 7500:
			marks += 0.75
			print("Total error is too high")
		elif err < 8000:
			marks += 0.5
		else:
			marks += 0
	except Exception as e:
		print("Error in task 5")
		print(e)
	try:
		X, Y = read_data('./data/kernel_design.csv')
		indices = np.arange(len(X))
		np.random.shuffle(indices)
		X = X[indices]
		Y = Y[indices]
		X_train = X[:600,:]
		Y_train = Y[:600,:]
		X_test = X[600:,:]
		Y_test = Y[600:,:]
		clf = KernelRidgeRegression(my_kernel,0.01,0.1)
		clf.fit(X_train, Y_train)
		# plot_3D(X[:,0], X[:,1], Y)
		clf = KernelRidgeRegression(my_kernel,0.01,0.1)
		clf.fit(X, Y)

		err = np.linalg.norm(clf.predict(X_test)-Y_test)**2
		# plt.show()
		if err < 2200:
			marks += 0.5
		elif err < 2700:
			marks += 0.25
			print("Test error is too high")
		else:
			print("Test error is too high")
			marks += 0
	except Exception as e:
		print("Error in task 5")
		print(e)		
	print(f"Marks obtained {marks}")
	return marks


print(f'Total Marks = {grade5()+grade4()+grade3()+grade2()+grade1()}')