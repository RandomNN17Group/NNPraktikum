# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer

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
    layer : LogisticLayer
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

        # Initialize layer
        self.layer = LogisticLayer(self.trainingSet.input.shape[1], 1, None,'sigmoid', True)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0

        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                output = self.fire(input)
                # compute gradient
                grad = label - output

                # compute recognizing error, not BCE
                predictedLabel = self.classify(input)
                error = loss.calculateError(label, predictedLabel)
                totalError += error

                self.updateWeights(grad)

            totalError = abs(totalError)
            
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)
                

            if totalError == 0 or iteration >= self.epochs:
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
        return self.fire(testInstance) > 0.5

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
        self.layer.computeDerivative(grad, None)
        self.layer.updateWeights(self.learningRate)

    def fire(self, input):
        return self.layer.forward(np.insert(input, 0, 1))
