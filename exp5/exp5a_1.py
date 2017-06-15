import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys

# Download data from .mat file into numpy array
print('==> Experiment 5a')
filepath = 'exp1l_C1.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_test = np.array(f.get('validationFeatures'))
y_test = np.array(f.get('validationLabels'))
del f
print('==> Data sizes:',X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Transform labels into on-hot encoding form
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.copy()).astype(int).toarray()
y_test = enc.fit_transform(y_test.copy()).astype(int).toarray()

# cubic root
X_train = np.cbrt(X_train)
X_test = np.cbrt(X_test)

# normalize X_train and X_test
X_train_squared = np.square(X_train)
sumOfSquares = X_train_squared.sum(axis=1)
X_train = X_train / np.sqrt(sumOfSquares[:, np.newaxis]) # divide each row by its magnitude

X_test_squared = np.square(X_test)
sumOfSquares = X_test_squared.sum(axis=1)
X_test = X_test / np.sqrt(sumOfSquares[:, np.newaxis])

# Functions for initializing neural nets parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

'''
    NN config parameters
'''
num_features = 169
hidden_size_list = [300]
num_classes = y_test.shape[1]
print("Number of features:", num_features)
print("Number of songs:",num_classes)
plotx = []
ploty_train = []
ploty_val = []

for j in range(len(hidden_size_list)):

    hidden_layer_size = hidden_size_list[j]
    plot_error_train = []
    plot_error_val = []
        
    # Set-up NN layers
    x = tf.placeholder(tf.float64, [None, num_features])
    W1 = weight_variable([num_features, hidden_layer_size])
    b1 = bias_variable([hidden_layer_size])

    # Hidden layer activation function: ReLU
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = weight_variable([hidden_layer_size, num_classes])
    b2 = bias_variable([num_classes])
    # Softmax layer (Output), dtype = float64
    y = tf.matmul(h1, W2) + b2

    # NN desired value (labels)
    y_ = tf.placeholder(tf.float64, [None, num_classes])

    # Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    sess.run(tf.global_variables_initializer())

    # Training
    numTrainingVec = len(X_train)
    batchSize = 500
    numEpochs = 1000

    startTime = time.time()
    for epoch in range(numEpochs):
        for i in range(0,numTrainingVec,batchSize):

            # Batch Data
            batchEndPoint = min(i+batchSize, numTrainingVec)
            trainBatchData = X_train[i:batchEndPoint]
            trainBatchLabel = y_train[i:batchEndPoint]

            train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

        # Print accuracy
        if epoch%50 == 0 or epoch == numEpochs-1:
            if j == 0:
                plotx.append(epoch)
            train_error = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
            validation_error = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})
            plot_error_train.append(train_error)
            plot_error_val.append(validation_error)
            print("epoch: %d, train error %g, val error %g"%(epoch, train_error, validation_error))

    endTime = time.time()
    print("Elapse Time:", endTime - startTime)

    ploty_train.append(plot_error_train)
    ploty_val.append(plot_error_val)

    # Validation

    train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
    validation_accuracy = accuracy.eval(feed_dict={x:X_test, y_: y_test})
    train_error = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
    validation_error = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})

    print("Train accuracy:",train_accuracy)
    print("Validation accuracy:",validation_accuracy)
    print("Train error:", train_error)
    print("Validation error:", validation_error)
    print("============================================")

print('==> Generating error plot...')
train_errfig = plt.figure()
hidden_100 = train_errfig.add_subplot(111)
hidden_100.set_xlabel('Number of Epochs')
hidden_100.set_ylabel('Cross-Entropy Error')
hidden_100.set_title('Error vs Number of Epochs')
hidden_100.scatter(plotx, ploty_train[0])
hidden_200 = train_errfig.add_subplot(111)
hidden_200.scatter(plotx, ploty_train[1], c='r')
hidden_300 = train_errfig.add_subplot(111)
hidden_300.scatter(plotx, ploty_train[2], c='g')
hidden_400 = train_errfig.add_subplot(111)
hidden_400.scatter(plotx, ploty_train[3], c='yellow')
train_errfig.savefig('exp5a_train_error.png')

val_errfig = plt.figure()
hidden_100 = val_errfig.add_subplot(111)
hidden_100.set_xlabel('Number of Epochs')
hidden_100.set_ylabel('Cross-Entropy Error')
hidden_100.set_title('Error vs Number of Epochs')
hidden_100.scatter(plotx, ploty_val[0])
hidden_200 = val_errfig.add_subplot(111)
hidden_200.scatter(plotx, ploty_val[1], c='r')
hidden_300 = val_errfig.add_subplot(111)
hidden_300.scatter(plotx, ploty_val[2], c='g')
hidden_400 = val_errfig.add_subplot(111)
hidden_400.scatter(plotx, ploty_val[3], c='yellow')
val_errfig.savefig('exp5a_val_error.png')
