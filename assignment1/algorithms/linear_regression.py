"""
Linear Regression model
"""

import numpy as np

class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None #Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay


    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
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
        return X_test.dot(self.w.T).argmax(axis=1)
        