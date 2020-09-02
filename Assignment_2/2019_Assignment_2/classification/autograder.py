import optparse
import os
import re
import sys

import perceptron1vr
import samples
import util
import dataClassifier

def runTask(task):
	print ("Grading task "+str(task) )
	if task == 2 or task == 5 :
		print ("This is a manually graded task, write your answers in the pdf file")
	elif task == 1: 
		print ("The solution cannot be exposed to you now. :)")
	elif task ==3:
		print("Ungraded task")

	else :
		if task == 6:
			numTraining = 800
			numTest = 200
			num_classes = 4
		else:
			numTraining = dataClassifier.TRAIN_SET_SIZE
			numTest = dataClassifier.TEST_SET_SIZE 
			num_classes = 10
			
		if task == 6:
			rawTrainingData = samples.loadDataFile("data/D2/training_data", numTraining)
			trainingLabels = samples.loadLabelsFile("data/D2/training_labels", numTraining)
			rawTestData = samples.loadDataFile("data/D2/test_data", numTest)
			testLabels = samples.loadLabelsFile("data/D2/test_labels", numTest)
			featureFunction = dataClassifier.enhancedFeatureExtractorDigit

		else:
			rawTrainingData = samples.loadDataFile("data/D1/training_data", numTraining)
			trainingLabels = samples.loadLabelsFile("data/D1/training_labels", numTraining)
			rawTestData = samples.loadDataFile("data/D1/test_data", numTest)
			testLabels = samples.loadLabelsFile("data/D1/test_labels", numTest)
			featureFunction = dataClassifier.basicFeatureExtractorDigit
		
		legalLabels = range(num_classes)
		classifier = perceptron1vr.Perceptron1vrClassifier(legalLabels,3)

		# Extract features
		print ("Extracting features...")
		trainingData = map(featureFunction, rawTrainingData)
		testData = map(featureFunction, rawTestData)

		# Conduct training and testing
		print ("Training...")
		classifier.train(trainingData, trainingLabels, testData, testLabels, False)

		print ("Testing...")
		guesses = classifier.classify(testData)
		correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
		acc = 100*correct/(1.0*(len(testLabels)))
		
		if task == 4:
			marks = 0
			if (acc > 70):
				marks = 3
			elif (acc >  60):
				marks = 2
			elif (acc >  50):
				marks = 1
			print ("Received Marks : " + str(marks) +"/3")
		elif task == 6:
			marks = 0
			print(acc)
			if len(testData[0]) <= 5 :
				if (acc > 85):
					marks = 3
				elif (acc > 65):
					marks = 2
				elif (acc >  45):
					marks = 1
			else :
				print ("More than permissible features used")
			print ("Received Marks : " + str(marks) +"/3")
	print ("--------------------------------------------------------"		)

if __name__ == '__main__':
	# Read input
	from optparse import OptionParser
	parser = OptionParser(dataClassifier.USAGE_STRING)
	parser.add_option('-t', '--task', help=dataClassifier.default('The task to autograde'), choices=['1', '2', '3', '4'], default=None)
	
	options, otherjunk = parser.parse_args(sys.argv[1:])
	if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

	task_list = [1, 2, 4, 5, 6]
	if options.task:
		runTask(int(options.task))
	else:
		for task in task_list:
			runTask(task)
			try :
				runTask(task)
			except:
				print ("Task" +str(task)	+" exited without running to completion")
				print ("Received Marks : " + str(0) +"/3")
				print ("--------------------------------------------------------")

