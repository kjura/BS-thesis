# Adaline model of singel-layer neural network

import numpy as np
import matplotlib.pyplot as plt
from rashka_chp2.iris_analysis import load_irisdata, get_setosa_versicolor
from rashka_chp2.raschka_code import AdalineGD

class Pydaline():
    """
    Implementation of ADALINE (ADAptive LInear NEuron), a model of single-layer NN.

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
        pseudo_generator = np.random.RandomState(self.rand_seed)
        self.w_ = pseudo_generator.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])

        # Create a list for cost function output
        # in case of knowing how well a network performs

        self.cost_ = []


        for i in range(self.epochs):

            #net_input_sum = self.calculate_net_input(X)
            #activation = self.calculate_activation_fun(net_input_sum)
            #error = y - net_input_sum
            #self.w_j[1:] += self.eta * X.T.dot(error)
            #self.w_j[0] += self.eta * np.sum(error)
            #SSE = (error**2).sum() / 2.0
            #self.cost_.append(SSE)
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """
        Calculate net input of weights and the input data,
        by calculating a dot product between two vectors.

        X : {array-like}, shape = (n_examples, n_features)
            Array containing vectorized data, where
            n_examples is the number of traning examples
            and n_features is the number of features.
        """
        # Note for myself
        # np.dot SOMETIMES is not commutative e.g
        # np.dot(X, self.w_j[1:]) != np.dot(self.w_j[1:], X)
        #
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        The identity function that outputs the input

        w_dot_X : {array-like}, shape = (n_examples, n_features)
                Array containing vectorized data, where
                n_examples is the number of traning examples
                and n_features is the number of features.
        """
        return X

    def predict(self, X):
        """
        The Heaviside step function, a threshold
        used to classify given class label.

        w_dot_X : {array-like}, shape = (n_examples, n_features)
                Array containing vectorized data, where
                n_examples is the number of traning examples
                and n_features is the number of features.

        Returns

        ---------

        out: ndarray
            An array with predicted outcomes.

        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


data = get_setosa_versicolor(load_irisdata())





def plot_cost_betw_2_models(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    network_A = Pydaline(eta=0.01, epochs=10, rand_seed=1).fit(X, y)
    network_B = Pydaline(eta=0.0001, epochs=10, rand_seed=1).fit(X, y)
    #network_A = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    #network_B = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)

    ax[0].plot(range(1, len(network_A.cost_) + 1), np.log10(network_A.cost_),  marker="o")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log_10(Sum squared error)")
    ax[0].set_title(f"Cost of Adaline with eta = {network_A.eta}")

    ax[1].plot(range(1, len(network_B.cost_) + 1), network_B.cost_,  marker="o")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Sum squared error")
    ax[1].set_title(f"Cost of Adaline with eta = {network_B.eta}")
    plt.show()





plot_cost_betw_2_models(data[0], data[1])