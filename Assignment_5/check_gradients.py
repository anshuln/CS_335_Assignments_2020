import numpy as np
import nn
import sys

from util import *
from layers import *

np.random.seed(42)

XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
XTrain = XTrain[0:10, :, :, :]
YTrain = YTrain[0:10, :]

nn1 = nn.NeuralNetwork(10, 1)

nn1.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 4, 1, 'relu'))
nn1.addLayer(MaxPoolingLayer([4, 30, 30], [3, 3], 3))
# nn1.addLayer(AvgPoolingLayer([4, 30, 30], [2, 2], 2))
# nn1.addLayer(ConvolutionLayer([4, 15, 15], [4, 4], 4, 1, 'relu'))
# nn1.addLayer(MaxPoolingLayer([4, 12, 12], [2, 2], 2))
nn1.addLayer(FlattenLayer())
nn1.addLayer(FullyConnectedLayer(400, 10, 'softmax'))

delta = 1e-7
r = nn1.layers[0].weights.shape[2]
c = nn1.layers[0].weights.shape[3]
num_grad = np.zeros((r, c))

for i in range(r):
	for j in range(c):
		activations = nn1.feedforward(XTrain)
		loss1 = nn1.computeLoss(YTrain, activations)
		nn1.layers[0].weights[0, 0, i, j] += delta
		activations = nn1.feedforward(XTrain)
		loss2 = nn1.computeLoss(YTrain, activations)
		num_grad_ij = (loss2 - loss1) / delta
		num_grad[i, j] = num_grad_ij
		nn1.layers[0].weights[0, 0, i, j] -= delta

saved = nn1.layers[0].weights[0, 0, :, :].copy()
activations = nn1.feedforward(XTrain)
nn1.backpropagate(activations, YTrain)
new = nn1.layers[0].weights[0, 0, :, :]
ana_grad = saved - new

assert np.linalg.norm(num_grad - ana_grad) < 1e-6
print("Gradient Test Passed!")