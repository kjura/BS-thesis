# Perceptron classifier in python
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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
    def __init__(self, eta, epoch=20, model_seed=1):
        self.eta = eta
        self.epoch = epoch
        self.model_seed = model_seed

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
                error_measure += 0 if delta_w_j == 0 else 1
            self.error.append(error_measure)
        return self

    def net_inp_fun(self, X):
        return np.dot(X, self.w_j[1:]) + self.w_j[0]

    def threshold_fun(self, X):
        return np.where(self.net_inp_fun(X) >= 0, 1.0, -1.0)

    def plot_error_of_updates(self,):
        plt.plot(range(1, len(self.error) + 1), self.error, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # Set up colors and markers for a Colormap plot as tuples
    # declare a Colormap , with as many colors as class labels
    colors = ("red", "green", "gold", "darkblue", "aquamarine")
    markers = ("^", "x", "+", "8", "o")
    cmap = ListedColormap(colors[:len(np.unique(y))])


    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Prepare a plot surface to divide class labels

    xv, yv = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    predicted_classlbl = classifier.threshold_fun(np.array([xv.ravel(), yv.ravel()]).T)
    predicted_classlbl = predicted_classlbl.reshape(xv.shape)


    plt.contourf(xv, yv, predicted_classlbl, alpha=0.3, cmap=cmap)
    plt.xlim(xv.min(), xv.max())
    plt.ylim(yv.min(), yv.max())


    # plot classified values

    for index, cls in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cls, 0], y=X[y == cls, 1], alpha=0.8,
                    c=colors[index], marker=markers[index], label=cls, edgecolors='black')


# page 27 and 41