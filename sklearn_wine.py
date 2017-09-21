import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import mglearn
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=50)

dfx = pd.read_csv('winequality.txt',
                  sep=';', usecols=range(11))
dfy = pd.read_csv('winequality.txt',
                  sep=';', usecols=range(11, 12))

X_train, X_test, y_train, y_test = train_test_split(
    dfx.values, dfy.values, random_state=0)

# iris_dataframe = pd.DataFrame(X_train, columns=dfx.columns.values)
# grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(
#     15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# plt.show()


knn.fit(X_train, y_train.ravel())


print("KNN set score: {:.2f}".format(
    knn.score(X_test, y_test.ravel())))
