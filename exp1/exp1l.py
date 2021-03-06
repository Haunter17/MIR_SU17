import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import time

# Download data from .mat file into numpy array
print('==> Experiment 1l')
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

# Functions for initializing neural nets parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

def setFreqRange(M, tb, numbins, totalFeatures):
    # Dimension of M is numSamples x 121
    if tb == 't':
        M = [x[0:totalFeatures-numbins] for x in M]
    elif tb == 'b':
        M = [x[numbins:totalFeatures] for x in M]
    else:
        M = [x[numbins:totalFeatures] for x in M]
    return M

removeBinsNum = [0, 13, 25, 37, 48]

for numbins in removeBinsNum:

    print("Number of Bins:",169-numbins)

    # Narrow frequency range
    tb = 'b'
    totalFeatures = 169
    X_train_sub = setFreqRange(X_train, tb, numbins, totalFeatures)
    X_test_sub = setFreqRange(X_test, tb, numbins, totalFeatures)

    '''
        NN config parameters
    '''
    num_features = totalFeatures - numbins
    hidden_layer_size = 100 # set according to exp1b
    num_classes = y_test.shape[1]
    print("Number of features:", num_features)
    print("Number of songs:",num_classes)

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
    numTrainingVec = len(X_train_sub)
    batchSize = 1000
    numEpochs = 500

    startTime = time.time()
    for epoch in range(numEpochs):
        for i in range(0,numTrainingVec,batchSize):

            # Batch Data
            batchEndPoint = min(i+batchSize, numTrainingVec)
            trainBatchData = X_train_sub[i:batchEndPoint]
            trainBatchLabel = y_train[i:batchEndPoint]

            train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

        # Print accuracy
        if (epoch%10 == 0 or epoch == numEpochs-1)  and noisy:
            train_accuracy = accuracy.eval(feed_dict={x:X_train_sub, y_: y_train})
            print("epoch: %d, training accuracy %g"%(epoch, train_accuracy))

    endTime = time.time()
    print("Elapse Time:", endTime - startTime)

    # Validation

    train_accuracy = accuracy.eval(feed_dict={x:X_train_sub, y_: y_train})
    validation_accuracy = accuracy.eval(feed_dict={x:X_test_sub, y_: y_test})
    train_error = cross_entropy.eval(feed_dict={x:X_train_sub, y_: y_train})
    validation_error = cross_entropy.eval(feed_dict={x:X_test_sub, y_: y_test})

    print("Train accuracy:",train_accuracy)
    print("Validation accuracy:",validation_accuracy)
    print("Train error:", train_error)
    print("Validation error:", validation_error)
    print("============================================")
