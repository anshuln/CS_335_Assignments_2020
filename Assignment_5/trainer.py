'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import nn
import numpy as np
import sys
import pickle

from util import *
from layers import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 64
			self.epochs = 20
			# self.lr = 0.001
			self.nn = nn.NeuralNetwork(out_nodes=10,lr=0.001)    
			self.nn.addLayer(FullyConnectedLayer(784,10,activation='softmax'))


		if dataset_name == 'CIFAR10':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]

			self.save_model = True
			self.model_name = "model.npy"

			# Add your network topology along with other hyperparameters here
			self.nn = nn.NeuralNetwork(10, 0.001)

			self.nn.addLayer(ConvolutionLayer(in_channels=(3,32,32),filter_size=(3,3),numfilters=6,stride=3,activation='relu'))
			# nn1.addLayer(AvgPoolingLayer(in_channels=(8,10,10),filter_size=(3,3),stride=3))
			# nn1.addLayer(ConvolutionLayer(in_channels=8,filter_size=3,numfilters=8,stride=3,activation='relu'))
			self.nn.addLayer(FlattenLayer())

			self.nn.addLayer(FullyConnectedLayer(600,10,activation='softmax'))
			self.batch_size =  64
			self.epochs = 20
			# self.lr = 
			# self.nn = 
			# nn.addLayer()

		if dataset_name == 'XOR':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()

			# self.save_model = True
			self.model_name = "model.npy"

			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 15
			self.lr = 1e-3
			self.nn = nn.NeuralNetwork(out_nodes=2, lr=self.lr)
			# nn.addLayer()
			self.nn.addLayer(FullyConnectedLayer(2, 3, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(3, 2, 'softmax'))

		if dataset_name == 'square':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 20
			self.lr = 1e-3
			self.nn = nn.NeuralNetwork(out_nodes=2, lr=self.lr)
			# nn.addLayer()
			self.nn.addLayer(FullyConnectedLayer(2, 3, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(3, 2, 'softmax'))


		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 20
			self.lr = 1e-3
			self.nn = nn.NeuralNetwork(out_nodes=2, lr=self.lr)
			# nn.addLayer()
			self.nn.addLayer(FullyConnectedLayer(2, 2, 'relu'))
			self.nn.addLayer(FullyConnectedLayer(2, 2, 'softmax'))

	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# A Training Epoch
			if verbose:
				print("Epoch: ", epoch)

			# TODO
			# Shuffle the training data for the current epoch

			X = np.asarray(self.XTrain)
			Y = np.asarray(self.YTrain)
			perm = np.arange(X.shape[0])
			np.random.shuffle(perm)
			X = X[perm]
			Y = Y[perm]

			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0

			# Divide the training data into mini-batches
			numBatches = int(np.ceil(float(X.shape[0]) / self.batch_size))
			for batchNum in range(numBatches):
				# print(batchNum,"/",numBatches) # UNCOMMENT if model is taking too long to train
				XBatch = np.asarray(X[batchNum*self.batch_size: (batchNum+1)*self.batch_size])
				YBatch = np.asarray(Y[batchNum*self.batch_size: (batchNum+1)*self.batch_size])

				# Calculate the activations after the feedforward pass
				activations = self.nn.feedforward(XBatch)  

				# Compute the loss  
				loss = self.nn.computeLoss(YBatch, activations)
				trainLoss += loss
				# print(self.out_nodes)
				# Estimate the one-hot encoded predicted labels after the feedword pass
				predLabels = oneHotEncodeY(np.argmax(activations[-1], axis=1), self.nn.out_nodes)

				# Calculate the training accuracy for the current batch
				acc = self.nn.computeAccuracy(YBatch, predLabels)
				trainAcc += acc
				# Backpropagation Pass to adjust weights and biases of the neural network
				self.nn.backpropagate(activations, YBatch)

			# END TODO
			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			if verbose:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)
			
			if self.save_model:
				model = []
				for l in self.nn.layers:
					# print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and verbose:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)

