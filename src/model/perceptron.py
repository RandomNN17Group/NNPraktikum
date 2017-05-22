# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from report.evaluator import Evaluator

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.DEBUG,
					stream=sys.stdout)


class Perceptron(Classifier):
	"""
	A digit-7 recognizer based on perceptron algorithm

	Parameters
	----------
	train : list
	valid : list
	test : list
	learningRate : float
	epochs : positive int

	Attributes
	----------
	learningRate : float
	epochs : int
	trainingSet : list
	validationSet : list
	testSet : list
	weight : list
	"""
	def __init__(self, train, valid, test,
									learningRate=0.01, epochs=50):

		self.learningRate = learningRate
		self.epochs = epochs

		self.trainingSet = train
		self.validationSet = valid
		self.testSet = test

		# Initialize the weight vector with small random values
		# around 0 and0.1
		self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

	def train(self, verbose=True):
		"""Train the perceptron with the perceptron learning algorithm.

		Parameters
		----------
		verbose : boolean
			Print logging messages with validation accuracy if verbose is True.
			
        for each epoch one must train the perceptron
        training equals classifying each element of the training set i.e. each input as one image/number#
        trainingSet contains an amount of training images=input and labels for each input
        training images contain an amount of pixels
        a pixel is a single input to the perceptron
        
        for correctly recognized 7, do nothing
        for incorrectly recognizing a number as 7, decrease weights
        for incorrectly not recognizing a number as 7, increase weights
		"""

		# Write your code to train the perceptron here

		for epoch in range(0, self.epochs):
			falseClassifications = 0.0
			for (trainingImage, label) in zip(self.trainingSet.input, self.trainingSet.label):
				classification = self.classify(trainingImage) #1/true if classified as 7, 0/false otherwise
				error = (label - classification) #target output/label and perceptron output/classification
				if(verbose & error != 0):
					falseClassifications += 1.0
				self.updateWeights(trainingImage, error)
			if verbose:
				accuracy = (1.0 - falseClassifications/float(len(self.trainingSet.input)))
				print("Epoch %d had accuracy of %f with %d false classifications" % (epoch, accuracy, falseClassifications))
				percEval = self.evaluate(self.validationSet)
				Evaluator().printAccuracy(self.validationSet, percEval)
		pass

	def classify(self, testInstance):
		"""Classify a single instance.

		Parameters
		----------
		testInstance : list of floats

		Returns
		-------
		bool :
			True if the testInstance is recognized as a 7, False otherwise.
		"""
		# Write your code to do the classification on an input image
		return self.fire(testInstance)
		pass

	def evaluate(self, test=None):
		"""Evaluate a whole dataset.

		Parameters
		----------
		test : the dataset to be classified
		if no test data, the test set associated to the classifier will be used

		Returns
		-------
		List:
			List of classified decisions for the dataset's entries.
		"""
		if test is None:
			test = self.testSet.input
		# Once you can classify an instance, just use map for all of the test
		# set.
		return list(map(self.classify, test))

	def updateWeights(self, input, error):
		# Write your code to update the weights of the perceptron here
		self.weight += self.learningRate*error*input
		pass

	def fire(self, input):
		"""Fire the output of the perceptron corresponding to the input """
		return Activation.sign(np.dot(np.array(input), self.weight))
