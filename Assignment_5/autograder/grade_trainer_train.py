import io
from contextlib import redirect_stdout

import numpy as np

import nn, nn_ref
from util import *
import layers_ref
import layers
from trainer import Trainer

def test_trainer_train():
    marks = 0

    trainer = Trainer('XOR')

    np.random.seed(337)

    trainer.XTrain, trainer.YTrain, _, _, trainer.XTest, trainer.YTest = readXOR()
    trainer.batch_size = 10
    trainer.epochs = 15
    trainer.lr = 1e-3
    trainer.nn = nn_ref.NeuralNetwork(out_nodes=2, lr=trainer.lr)
    trainer.nn.addLayer(layers_ref.FullyConnectedLayer(2, 3, 'relu'))
    trainer.nn.addLayer(layers_ref.FullyConnectedLayer(3, 2, 'softmax'))

    f = io.StringIO()
    with redirect_stdout(f):
        trainer.train(verbose=False)

    out = f.getvalue()

    acc = float(out.strip().split(' ')[-1]) # 95.6

    if acc > 94.5:
        marks += 2
    elif acc > 90:
        marks += 1

    return marks

print(test_trainer_train())

# def train(nn, XTrain, YTrain, XTest, YTest, epochs, batch_size, verbose=True):
# 	# Method for training the Neural Network
	
# 	# The methods trains the weights and baises using the training data(trainX, trainY)
# 	# and evaluates the validation set accuracy after each epoch of training

# 	for epoch in range(epochs):
# 		# Shuffle the training data for the current epoch
# 		X = np.asarray(self.XTrain)
# 		Y = np.asarray(self.YTrain)
# 		perm = np.arange(X.shape[0])
# 		np.random.shuffle(perm)
# 		X = X[perm]
# 		Y = Y[perm]

# 		# Initializing training loss and accuracy
# 		trainLoss = 0
# 		trainAcc = 0

# 		# Divide the training data into mini-batches
# 		numBatches = int(np.ceil(float(X.shape[0]) / batch_size))
# 		for batchNum in range(numBatches):
# 			XBatch = np.asarray(X[batchNum*batch_size: (batchNum+1)*batch_size])
# 			YBatch = np.asarray(Y[batchNum*batch_size: (batchNum+1)*batch_size])

# 			# Calculate the activations after the feedforward pass
# 			activations = nn.feedforward(XBatch)  

# 			# Compute the loss  
# 			loss = nn.computeLoss(YBatch, activations)
# 			trainLoss += loss
# 			# Estimate the one-hot encoded predicted labels after the feedword pass
# 			predLabels = oneHotEncodeY(np.argmax(activations[-1], axis=1), nn.out_nodes)

# 			# Calculate the training accuracy for the current batch
# 			acc = nn.computeAccuracy(YBatch, predLabels)
# 			trainAcc += acc
# 			# Backpropagation Pass to adjust weights and biases of the neural network
# 			nn.backpropagate(activations, YBatch)

# 		# Print Training loss and accuracy statistics
# 		trainAcc /= numBatches
		
# 	pred, acc = nn.validate(XTest, YTest)
# 	return acc
