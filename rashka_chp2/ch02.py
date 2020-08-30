# Perceptron classifier in python
import numpy as np


class Pyceptron():
    """
    My own implementation of a perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate given between 0.0 to 1.0
    epoch : int
        Maximum number of passes over the training dataset
    model_seed : int
        Seed for generating random weight initialization
    """
    def __init__(self, eta, epoch=50, model_seed=1):
        self.eta = eta
        self.epoch = epoch
        self.model_seed = model_seed

    def calc_error(self, predicted_val, true_val):
        return 0 if true_val - predicted_val == 0 else 1

    def fit(self, X, y):
        """
        Method that fits the model given the X-elements
        to y.
        :param X: Data for training
        :param y: Target values
        :return: self
        """

        self.error = []
        # Call for Mersenne Twister pseudo-random number generator
        pseudo_generator = np.random.RandomState(self.model_seed)
        # X.shape[1] gives us columns, .shape gives us a tuple (n, m)
        # n stands for rows and m for columns
        self.w_j = pseudo_generator.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])

        for epochs in range(self.epoch):
            error_measure = 0
            for training_example, true_class_lbl in zip(X, y):
                delta_w_j = self.eta * (true_class_lbl - self.threshold_fun(training_example))
                self.w_j[0] += delta_w_j * 1  # w_0 is multiplied by 1 where 1 is x_0
                self.w_j[1:] += delta_w_j * training_example
                error_measure += self.calc_error(self.threshold_fun(training_example), true_class_lbl)
            self.error.append(error_measure)
        return self

    def net_inp_fun(self, X):
        return np.dot(X, self.w_j[1:]) + self.w_j[0]

    def threshold_fun(self, X):
        return np.where(self.net_inp_fun(X) >= 0, 1.0, -1.0)

