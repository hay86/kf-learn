import sys
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from src import LogisticRegression as MyLogisticRegression
from src import LinearRegression as MyLinearRegression

if len(sys.argv) < 2:
	print 'Usage: python vs.py regression|classification'
	sys.exit()

if sys.argv[1] == 'regression':
	X, Y = make_regression(n_samples=100,n_features=5)
	m1 = LinearRegression()
	m2 = MyLinearRegression()
	W = [0] * (len(X[0])+1)
elif sys.argv[1] == 'classification':
	X, Y = make_classification(n_samples=100,n_features=5)
	m1 = LogisticRegression()
	m2 = MyLogisticRegression()
	W = [0] * len(X[0])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

# lr by sklearn
m1.fit(X_train, Y_train)
print '== test on sklearn:', m1.score(X_test,Y_test)

# lr by kflearn
W = m2.fit(X_train, Y_train, W, 0.1, 1000)
print '== test on kflearn:', m2.score(X_test,Y_test, W)

