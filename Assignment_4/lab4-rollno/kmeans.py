import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5927)

class KMeans():
	def __init__(self, D, n_clusters):
		self.n_clusters = n_clusters
		self.cluster_centers = np.zeros((n_clusters, D))

	def init_clusters(self, data):
		### TODO
		### Initialize cluster_centers using n_clusters points sampled from data

		### END TODO

	def pred(self, x):
		### TODO: Given a sample x, return id of closest cluster center

		### END TODO

	def train(self, data, max_iter):
		for _ in range(max_iter):
			### TODO
			### Declare and initialize required variables

			### Update labels for each point

			### Update cluster centers
			### Note: If some cluster is empty, do not update the cluster center

			### END TODO

	def replace_by_center(self, data):
		out = np.zeros_like(data)
		for i, x in enumerate(data):
			out[i] = self.cluster_centers[self.pred(x)]
		return out

if __name__ == '__main__':
	image = plt.imread('data/1.png')
	x = image.reshape(-1, 3)
	kmeans = KMeans(D=3, n_clusters=10)
	kmeans.init_clusters(x)
	kmeans.train(x, 5)
	out = kmeans.replace_by_center(x)
	plt.imshow(out.reshape(image.shape))
	plt.show()

