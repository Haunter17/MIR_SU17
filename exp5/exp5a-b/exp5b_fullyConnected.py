import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Download data from .mat file into numpy array
print('==> Experiment 5b_FullyConnected')
filepath = 'exp5b_taylorswift_d7_1s_C1C8.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))
window_X_train = np.array(f.get('window_trainFeatures'))
window_y_train = np.array(f.get('window_trainLabels'))
window_X_test = np.array(f.get('window_testFeatures'))
window_y_test = np.array(f.get('window_testLabels'))
del f
print('==> Data sizes:',X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# Transform labels into on-hot encoding form
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.copy()).astype(int).toarray()
y_val = enc.fit_transform(y_val.copy()).astype(int).toarray()
window_y_train = enc.fit_transform(window_y_train.copy()).astype(int).toarray()
window_y_test = enc.fit_transform(window_y_test.copy()).astype(int).toarray()

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
num_frames = 34
hidden_layer_size = 800
num_classes = y_val.shape[1]
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

# Each row = 1 window
y_window_flat = tf.reshape(y, [-1, num_classes*num_frames])

# New sets of weight independent of previous training
x_window = tf.placeholder(tf.float64, [None, num_classes*num_frames])
W3 = weight_variable([num_classes*num_frames, num_classes])
b3 = bias_variable([num_classes])
y_window = tf.matmul(x_window, W3) + b3

# NN desired value (labels)
y_ = tf.placeholder(tf.float64, [None, num_classes])

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
cross_entropy_window = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_window))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step_window = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_window)

sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction_window = tf.equal(tf.argmax(y_window, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
accuracy_window = tf.reduce_mean(tf.cast(correct_prediction_window, tf.float64))
sess.run(tf.global_variables_initializer())

# Training 1: Single Frame
numTrainingVec = len(X_train)
batchSize = 500
numEpochs = 250

noisy = False
startTime = time.time()
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, numTrainingVec)
        trainBatchData = X_train[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

endTime = time.time()
print("Finished Training 1 - Elapse Time:", endTime - startTime)

# Validation Part 1

train_error = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
validation_error = cross_entropy.eval(feed_dict={x:X_val, y_: y_val})
train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train})
val_acc = accuracy.eval(feed_dict={x:X_val, y_: y_val})

print("Train accuracy:",train_acc)
print("Validation accuracy:",val_acc)
print("Train error:", train_error)
print("Validation error:", validation_error)
print("============================================")

# Training 2: Fully Connected for Window
# Preprocess test data with trained MLP
def reshape_window(A, frames, features):
    A_byFrame = np.reshape(A,(-1, frames))
    A_trans = np.transpose(A_byFrame)
    A_byFeatures = np.reshape(A_trans, (-1, features))
    return A_byFeatures

window_X_train = reshape_window(window_X_train[:len(window_X_train)/2], num_frames, num_features)
window_X_test = reshape_window(window_X_test[:len(window_X_test)/2], num_frames, num_features)
window_y_train = window_y_train[:len(window_y_train)/2]
window_y_test = window_y_test[:len(window_y_test)/2]
print(window_X_train.shape, window_X_test.shape)

window_X_train_processed = y_window_flat.eval(feed_dict={x:window_X_train})
window_X_test_processed = y_window_flat.eval(feed_dict={x:window_X_test})
print(window_X_train_processed.shape, window_X_test_processed.shape)

num_train_window_vec = len(window_X_train_processed)

plotx = []
ploty_train = []
ploty_val = []
numEpochs = 2000

startTime = time.time()
for epoch in range(numEpochs):
    for i in range(0, num_train_window_vec, batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, num_train_window_vec)
        trainBatchData = window_X_train_processed[i:batchEndPoint]
        trainBatchLabel = window_y_train[i:batchEndPoint]

        train_step_window.run(feed_dict={x_window: trainBatchData, y_: trainBatchLabel})

    # Print accuracy
    if (epoch%50 == 0 or epoch == numEpochs-1):
        plotx.append(epoch)
        train_error = cross_entropy_window.eval(feed_dict={x_window :window_X_train_processed, y_: window_y_train})
        validation_error = cross_entropy_window.eval(feed_dict={x_window:window_X_test_processed, y_: window_y_test})
        train_acc = accuracy_window.eval(feed_dict={x_window:window_X_train_processed, y_: window_y_train})
        val_acc = accuracy_window.eval(feed_dict={x_window:window_X_test_processed, y_: window_y_test})
        ploty_train.append(train_error)
        ploty_val.append(validation_error)
        print("epoch: %d, train acc %g, val acc %g, train error %g, val error %g"%(epoch, train_acc, val_acc, train_error, validation_error))

endTime = time.time()
print("Finished Training 2 - Elapse Time:", endTime - startTime)

# Validation

train_error = cross_entropy_window.eval(feed_dict={x_window:window_X_train_processed, y_: window_y_train})
validation_error = cross_entropy_window.eval(feed_dict={x_window:window_X_test_processed, y_: window_y_test})
train_acc = accuracy_window.eval(feed_dict={x_window:window_X_train_processed, y_: window_y_train})
val_acc = accuracy_window.eval(feed_dict={x_window:window_X_test_processed, y_: window_y_test})

print("Train accuracy:",train_acc)
print("Validation accuracy:",val_acc)
print("Train error:", train_error)
print("Validation error:", validation_error)
print("============================================")

print('==> Generating error plot...')
errfig = plt.figure()
trainErrPlot = errfig.add_subplot(111)
trainErrPlot.set_xlabel('Number of Epochs')
trainErrPlot.set_ylabel('Cross-Entropy Error')
trainErrPlot.set_title('Error vs Number of Epochs')
trainErrPlot.scatter(plotx, ploty_train)
valErrPlot = errfig.add_subplot(111)
valErrPlot.scatter(plotx, ploty_val, c='r')
errfig.savefig('exp5b_fc_error.png')