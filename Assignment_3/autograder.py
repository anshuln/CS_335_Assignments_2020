import numpy as np
from kmeans import KMeans

def gradex():
	marks = 0
	try:
		data = np.array([[i,i] for i in range(5)])
		centers = np.array([[1.,1.], [2.,2.], [3.,3.]])
		op = np.array([[0.5, 0.5], [2.0, 2.0], [3.5, 3.5]])

		kmeans = KMeans(D=2, n_clusters=3)
		kmeans.cluster_centers = centers
		kmeans.train(data, 1)
		if np.allclose(kmeans.cluster_centers, op):
			marks += 0.25

		data = np.array([[i+1,i*2.3] for i in range(5)])
		centers = np.array([[5.,1.], [-1.,2.], [3.,6.]])
		op = np.array([[5, 1], [1.5, 1.15], [4.0, 6.8999999999999995]])

		kmeans = KMeans(D=2, n_clusters=3)
		kmeans.cluster_centers = centers
		kmeans.train(data, 1)
		if np.allclose(kmeans.cluster_centers, op):
			marks += 0.25

		data = np.array([[i+1,i*2.3] for i in range(3)])
		centers = np.array([[5, 1], [-1., 2]])
		op = np.array([[3.0, 4.6], [1.5, 1.15]])
		kmeans = KMeans(D=2, n_clusters=2)
		kmeans.cluster_centers = centers
		kmeans.train(data, 5)
		if np.allclose(kmeans.cluster_centers, op):
			marks += 0.5
	except:
		print('Error in k-means')
	return marks

print(f'Total Marks = {gradex()}')