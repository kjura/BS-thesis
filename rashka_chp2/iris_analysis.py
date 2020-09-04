import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rashka_chp2.perceptron_model import Pyceptron, plot_decision_regions



def load_irisdata():
    df = pd.read_csv("../data/iris.csv", header=None, encoding="utf-8")
    return df


def get_setosa_versicolor(df):
    # Label Iris-setosa as -1 and iris-versicolor as 1
    y = np.where(df.iloc[0:100, 4].values == 'Iris-setosa', -1, 1)
    # get sepal vs petal length
    X = df.iloc[0:100, [0, 2]].values
    return X, y




def sct_setosa2versicolor(df):
    y = np.where(df.iloc[0:100, 4].values == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='^', label='versicolor')
    plt.tick_params(axis='x', colors='red')
    plt.tick_params(axis='y', colors='red')

    plt.xlabel('sepal length [cm]', color="red")
    plt.ylabel('petal length [cm]', color="red")
    plt.legend(loc='upper left')
    plt.show()
    return X, y


ppn = Pyceptron(eta=0.01, epoch=20, model_seed=39)


data = get_setosa_versicolor(load_irisdata())
ppn.fit(data[0], data[1])
#plot_decision_regions(data[0], data[1], classifier=ppn)
#plt.xlabel("sepal length [cm]")
#plt.ylabel("petal length [cm]")
#plt.legend(loc="upper left")
#plt.show()
ppn.plot_error_of_updates()

