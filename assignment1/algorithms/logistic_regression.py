"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5 # To threshold the sigmoid 
        self.weight_decay = weight_decay


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        
        sigma = np.zeros_like(z)
        sigma[z < 0] = 1 / (np.exp(self.n_class * z[z < 0]) + 1)
        sigma[z >= 0] = 1 / (1 + np.exp(-1 * self.n_class * z[z >= 0]))

        return sigma


    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        

        N, _ = X_train.shape
        self.w = weights
        dW = np.zeros_like(self.w)

        # One-hot
        y_train_hot = np.eye(self.n_class)[y_train]

        # The update formula
        for _ in range(self.epochs):
            for i in range(N):
                dW += 2 * np.outer((np.dot(self.w, X_train[i].T) - y_train_hot[i]), X_train[i].T)
            dW /= N
            self.w = self.w - (self.lr * ((self.weight_decay * self.w) + dW))


        return self.w


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        return self.sigmoid(X_test.dot(self.w.T)).argmax(axis=1)
