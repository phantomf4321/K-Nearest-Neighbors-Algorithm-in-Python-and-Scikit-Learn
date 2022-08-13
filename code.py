import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

dataset.head()

"""
	sepal-length 	sepal-width 	petal-length 	petal-width 	Class
0 	5.1 	3.5 	1.4 	0.2 	Iris-setosa
1 	4.9 	3.0 	1.4 	0.2 	Iris-setosa
2 	4.7 	3.2 	1.3 	0.2 	Iris-setosa
3 	4.6 	3.1 	1.5 	0.2 	Iris-setosa
4 	5.0 	3.6 	1.4 	0.2 	Iris-setosa

"""


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
#KNeighborsClassifier()


y_pred = classifier.predict(X_test)
#print(y_pred)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""
[[10  0  0]
 [ 0 11  0]
 [ 0  1  8]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       0.92      1.00      0.96        11
 Iris-virginica       1.00      0.89      0.94         9

       accuracy                           0.97        30
      macro avg       0.97      0.96      0.97        30
   weighted avg       0.97      0.97      0.97        30

"""
