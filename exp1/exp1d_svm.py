from sklearn.svm import SVC
from sklearn import cross_validation
import numpy as np
import h5py
import time

print('==> Experiment 1d')
filepath = '(separate features & labels) exp1a_smallDataset_71_7.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_test = np.array(f.get('validationFeatures'))
y_test = np.array(f.get('validationLabels'))
del f
print('==> Data sizes:',X_train.shape, y_train.shape, X_test.shape, y_test.shape)

y_train = [i[0] for i in y_train]
y_test = [i[0] for i in y_test]

clf = SVC()
startTime = time.time()
clf = clf.fit(X_train, y_train)
endTime = time.time()
print("Time:",endTime-startTime)
print("Score:",clf.score(X_test,y_test))
print("======================================")