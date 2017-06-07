# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Try to use the abstract way of the framework
        # from util.loss_functions import MeanSquaredError
        from util.loss_functions import DifferentError
        diff = DifferentError()
        # mean = MeanSquaredError()

        learned = False
        iteration = 0

        # Train for some epochs if the error is not 0
        while not learned:
            totalErrors = 0
            grad = np.zeros(self.weight.size)
            trainingOutput = list(map(self.fire, self.trainingSet.input))
            trainingClassified = self.classify(self.trainingSet.input)
            for input, label, output, classified in zip(self.trainingSet.input,
                                    self.trainingSet.label, trainingOutput, trainingClassified):
                error = (label - output) * Activation.sigmoidPrime(output)
                grad -= error * input
                totalErrors += np.abs(diff.calculateError(label, classified))

            self.updateWeights(grad)

            iteration += 1

            gradAbs = np.sum(np.abs(grad))

            if verbose:
                from report.evaluator import Evaluator
                # meanErrorOld = mean.calculateError(self.trainingSet.label, trainingClassified)
                # meanErrorNew = mean.calculateError(self.trainingSet.label, self.classify(self.trainingSet.input))
                # meanDifference = meanErrorOld - meanErrorNew
                # logging.info("Epoch: %i; False training classifications: %i; Abs of Gradient: %.20f; Mean Squared Error: %.50f%%", iteration, totalErrors, gradAbs, meanDifference)
                logging.info("Epoch: %i; False training classifications: %i; Abs of Gradient: %.20f", iteration, totalErrors, gradAbs)
                print "Validation",
                Evaluator().printAccuracy(self.validationSet, self.evaluate(self.validationSet))

            #Add "or gradAbs < epsilon" with a suited epsilon as aditional stop criteria
            if totalErrors == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True
        
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
        return Activation.sign(self.fire(testInstance), 0.5)

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

    def updateWeights(self, grad):
        self.weight -= self.learningRate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
