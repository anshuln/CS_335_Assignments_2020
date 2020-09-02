# Mira implementation
import util
PRINT = True

class MiraClassifier:
	"""
	Mira classifier.

	Note that the variable 'datum' in this code refers to a counter of features
	(not to a raw samples.Datum).
	"""
	def __init__( self, legalLabels, max_iterations):
		self.legalLabels = legalLabels
		self.type = "mira"
		self.automaticTuning = False # Look at this flag to decide whether to choose C automatically ** use this in your train method **
		self.C = 0.001
		self.max_iterations = max_iterations
		self.weights = {}
		for label in legalLabels:
			self.weights[label] = util.Counter() # this is the data-structure you should use

	def train(self, trainingData, trainingLabels, testData, testLabels, validate):
		"""
		Outside shell to call your method. Do not modify this method.
		"""  
		  
		self.features = trainingData[0].keys() # this could be useful for your code later...

		if (self.automaticTuning):
			Cgrid = [0.001, 0.002, 0.003, 0.004, 0.005]
		else:
			Cgrid = [self.C]
			
		return self.trainAndTune(trainingData, trainingLabels, testData, testLabels, Cgrid, validate)

	def initializeWeightsToZero(self):
		"""
		Reset the weights of each label to zero vectors
		"""
		## YOUR CODE BELOW
		
		util.raiseNotDefined()
		return

	def trainAndTune(self, trainingData, trainingLabels, testData, testLabels, Cgrid, validate):
		"""
		See the project description for details how to update weight vectors for each label in training step. 

		Use the provided self.weights[label] datastructure so that 
		the classify method works correctly. Also, recall that a
		datum is a counter from features to values for those features
		(and thus represents a vector a values).

		This method needs to return the best parameter found in the list of parameters Cgrid
		(i.e. the parameter that yeilds best accuracy for the validation dataset)
		"""

		selectedC = Cgrid[0]

		# YOUR CODE HERE (Initializations)
		
		# END

		f = open("miraIterations.csv","w")
		f_tr = open("miraIterationsTrain.csv", "w")
		for c in Cgrid:
			self.initializeWeightsToZero()
			for iteration in range(self.max_iterations):
				for i in range(len(trainingData)):
					if(validate):
						if (i % (len(trainingData)/20) == 0):
							guesses = self.classify(testData)
							correct = [guesses[j] == testLabels[j] for j in range(len(testLabels))].count(True)
							f.write(str(c)+','+str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(testData)))+"\n")
							# f.write(str(seen)+","+str(100*correct/(1.0*len(testData)))+"\n")
							guesses = self.classify(trainingData)
							correct = [guesses[j] == trainingLabels[j] for j in range(len(trainingLabels))].count(True)
							f_tr.write(str(c)+','+str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")

					## YOUR CODE BELOW
					

			## Do not edit code below				
			if(validate):
				guesses = self.classify(testData)
				correct = [guesses[j] == testLabels[j] for j in range(len(testLabels))].count(True)
				f.write(str(c)+','+str(self.max_iterations*len(trainingData))+","+str(100*correct/(1.0*len(testData)))+"\n")                        
				guesses = self.classify(trainingData)
				correct = [guesses[j] == trainingLabels[j] for j in range(len(trainingLabels))].count(True)
				f_tr.write(str(c)+','+str(self.max_iterations*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")
		
		self.weights = finalWeights

				
		f.close()
		f_tr.close()

		return selectedC


	def classify(self, data ):
		"""
		Classifies each datum as the label that most closely matches the prototype vector
		for that label.  See the project description for details.

		Recall that a datum is a util.counter... 
		"""
		guesses = []
		for datum in data:
			vectors = util.Counter()
			for l in self.legalLabels:
				vectors[l] = self.weights[l] * datum
			guesses.append(vectors.argMax())
		return guesses