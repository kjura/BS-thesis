# Adaline model of singel-layer neural network

import numpy as np
import matplotlib.pyplot as plt
from rashka_chp2.perceptron_model import Pyceptron
from rashka_chp2.iris_analysis import load_irisdata, get_setosa_versicolor

class Pydaline():
    """
    Implementation of ADALINE (ADAptive LInear NEuron), a model of sigle-layer NN.

    Parameters
    -----------
    eta : float
        Learning rate, a measure how quickly network can learn.
    epochs : int
        Maximum number of passes over the training dataset.
    rand_seed: int
        A seed necessary for initial weight randomization.
    """

    def __init__(self, eta, epochs, rand_seed):
        self.eta = eta
        self.epochs = epochs
        self.rand_seed = rand_seed

    # Call for Mersenne Twister pseudo-random number generator
    # Generate random weights
    def generate_rand_weights(self, X):
        pseudo_generator = np.random.RandomState(self.rand_seed)
        self.w_j = pseudo_generator.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        return self.w_j

    def fit(self, X, y):
        """
        Fit training data

        Parameters
        ----------

        X : {array-like}, shape = (n_examples, n_features)
            Array containing vectorized data, where
            n_examples is the number of traning examples
            and n_features is the number of features.
        y : {array-like}, shape = (n_examples)
            Target values for classification.

        Returns
        ---------
        self : Object

        """

        self.weights = self.generate_rand_weights()

        # Create a list for cost function output
        # in case of knowing how well a network performs

        self.cost_ = []


        def calculate_net_input(self, X):
            """
            Calculate net input of weights and the input data,
            by calculating a dot product between two vectors.

            X : {array-like}, shape = (n_examples, n_features)
                Array containing vectorized data, where
                n_examples is the number of traning examples
                and n_features is the number of features.
            """
            return np.dot(self.w_j[1:], X) + self.w_j[0]

        def calculate_activation_fun(self, w_dot_X):
            return w_dot_X
