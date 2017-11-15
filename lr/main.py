import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src import LogisticRegression as KFLR

if len(sys.argv) < 2:
	print 'Usage: python main.py test.csv'
	sys.exit()

df = pd.read_csv(sys.argv[1])
col = df.columns

X = df[col[1:-1]]
X = np.array(X)
Y = df['label']
Y = np.array(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

# lr by sklearn
clf = LogisticRegression()
clf.fit(X_train, Y_train)
print '== test on sklearn:', clf.score(X_test,Y_test)

# lr by kflearn
clf = KFLR()
W = [0] * len(X[0])
W = clf.fit(X_train, Y_train, W, 0.1, 1000)
print '== test on kflearn:', clf.score(X_test,Y_test, W)

