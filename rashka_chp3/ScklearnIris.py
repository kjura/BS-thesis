from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# structure of iris
#1. sepal length in cm index 0 of data
#2. sepal width in cm index 1 of data
#3. petal length in cm index 2 of data
#4. petal width in cm index 3 of data
#5. class:
#-- Iris Setosa
#-- Iris Versicolour
#-- Iris Virginica




X = iris.data[:, [2, 3]]
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y)