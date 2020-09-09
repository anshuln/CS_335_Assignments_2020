import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5927)

class KMeans():
	def __init__(self, D, n_clusters):
		self.n_clusters = n_clusters
		self.cluster_centers = np.zeros((n_clusters, D))

	def init_clusters(self, data):
		idx = np.random.randint(0, data.shape[0], self.n_clusters)
		self.cluster_centers = data[idx]

	def pred(self, x):
		dist = np.sum((self.cluster_centers - x.reshape(1, -1))**2, axis=1)
		return np.argmin(dist)

	def train(self, data, max_iter):
		for _ in range(max_iter):
			label = np.zeros(data.shape[0], dtype=int)
			count = np.zeros(self.n_clusters, dtype=int)

			for i, x in enumerate(data):
				label[i] = self.pred(x)
				count[label[i]] += 1

			for i in range(self.n_clusters):
				if count[i]>0:
					self.cluster_centers[i] = np.mean(data[label==i, :], axis=0)
				# else:
				# 	self.cluster_centers[i] = data[np.random.randint(data.shape[0])]

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

