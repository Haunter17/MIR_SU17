import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('==> Experiment 1b')
filepath = '../taylorswift_out/small.mat'
print('==> Loading data from {}'.format(filepath))
# benchmark
t_start = time.time()

# reading data
f = h5py.File(filepath)
data_train = np.array(f.get('trainingSet'))
X_train = data_train[:, :-1]
y_train = data_train[:, -1].reshape(-1, 1)
data_test = np.array(f.get('testSet'))
X_test = data_test[:, :-1]
y_test = data_test[:, -1].reshape(-1, 1)
t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
    seconds'.format(t = t_end - t_start))
del data_train, data_test, f
print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of test samples: {}'.format(X_test.shape[0]))

# Neural-network model set-up
# Functions for initializing neural nets parameters
def init_weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
  return tf.Variable(initial)

def init_bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
  return tf.Variable(initial)

'''
	NN config parameters
'''
num_featuers = 121
hidden_layer_size = 20
num_classes = int(max(y_train.max(), y_test.max()) + 1)

# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_test_OHEnc = tf.one_hot(y_test.copy(), num_classes)

# Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_featuers])
W1 = init_weight_variable([num_featuers, hidden_layer_size])
b1 = init_bias_variable([hidden_layer_size])

# Hidden layer activation function: ReLU
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = init_weight_variable([hidden_layer_size, num_classes])
b2 = init_bias_variable([num_classes])

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

y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_test = sess.run(y_test_OHEnc)[:, 0, :]

'''
	Training config
'''
numTrainingVec = len(X_train)
batchSize = 1000
numEpochs = 300
print_freq = 5

train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

# benchmark
t_start = time.time()
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i + batchSize, numTrainingVec)
        trainBatchData = X_train[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        # train
        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    # Print accuracy
    if (epoch + 1) % print_freq == 0:
        # evaluate accuracy and errors
        train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train})
        train_acc_list.append(train_acc)
        val_acc = accuracy.eval(feed_dict={x: X_test, y_: y_test})
        val_acc_list.append(val_acc)
        train_err = cross_entropy.eval(feed_dict={x: X_train, y_: y_train})
        train_err_list.append(train_err)
        val_err = cross_entropy.eval(feed_dict={x: X_test, y_: y_test})
        val_err_list.append(val_err)      
        print("-- epoch: %d, training accuracy %g"%(epoch + 1, train_acc))

t_end = time.time()
print('--Time elapsed for training: {t:.2f} \
    seconds'.format(t = t_end - t_start))

# Reports
print('-- Training accuracy: {:.4f}'.format(train_acc_list[-1]))
print('-- Validation accuracy: {:.4f}'.format(val_acc_list[-1]))
print('-- Training error: {:.4E}'.format(train_err_list[-1]))
print('-- Validation error: {:.4E}'.format(val_err_list[-1]))

# Generating plots


print('==> Generating accuracy plot...')
x_list = range(0, print_freq * len(train_acc_list), print_freq)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Number of Epochs')
train_acc_plot, = plt.plot(x_list, train_acc_list, 'bo')
val_acc_plot, = plt.plot(x_list, val_acc_list, 'ro')
plt.legend((train_acc_plot, val_acc_plot), ('training', 'validation'), loc='best')
plt.savefig('exp1b_accuracy.png', format='png')
plt.close()

print('==> Generating error plot...')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs')
train_err_plot, = plt.plot(x_list, train_err_list, 'bo')
val_err_plot, = plt.plot(x_list, val_err_list, 'ro')
plt.legend((train_err_plot, val_err_plot), ('training', 'validation'), loc='best')
plt.savefig('exp1b_error.png', format='png')
plt.close()

print('==> Done.')

