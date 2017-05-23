import h5py
import numpy as np
print('==> Experiment 1b')
filepath = '../taylorswift_out/data.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
data_train = np.array(f.get('trainingSet'))
X_train = data_train[:, :-1]
y_train = data_train[:, -1].reshape(-1, 1)
data_test = np.array(f.get('testSet'))
X_test = data_test[:, :-1]
y_test = data_test[:, -1].reshape(-1, 1)
del data_train, data_test, f
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
