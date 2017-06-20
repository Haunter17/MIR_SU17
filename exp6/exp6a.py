import h5py
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAST_FLAG = 1
print('==> Experiment 6a: LSTM memory sizes')
filepath = '/pylon2/ci560sp/haunter/exp3_taylorswift_d15_1s_C1C8.mat'
if FAST_FLAG:
    filepath = '/pylon2/ci560sp/haunter/exp3_small.mat'
print('==> Loading data from {}...'.format(filepath))
# benchmark
t_start = time.time()

# ==============================================
#               reading data
# ==============================================
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))

t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
        seconds'.format(t = t_end - t_start))
del f

print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of validation samples: {}'.format(X_val.shape[0]))


# ==============================================
#               RNN configs
# ==============================================
# Network Parameters
num_training_vec, total_features = X_train.shape
num_freq = 169
num_frames = int(total_features / num_freq)

max_iter = 300
print_freq = 10
if FAST_FLAG:
    max_iter = 10
    print_freq = 1

batch_size = 1000
learning_rate = 0.001
n_input = num_freq # number of sequences (rows)
n_steps = num_frames # size of each sequence (number of columns), timesteps
n_hidden = 128 # hidden layer num of features
n_classes = int(max(y_train.max(), y_val.max()) + 1)

try:
    n_hidden = int(sys.argv[1])
except Exception, e:
    print('-- {}'.format(e))
print('-- LSTM size: '.format(n_hidden))

# ==============================================
#               RNN architecture
# ==============================================
# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), n_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), n_classes)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# evaluation metrics
train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

# ==============================================
#               RNN training
# ==============================================
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    y_train = sess.run(y_train_OHEnc)[:, 0, :]
    y_val = sess.run(y_val_OHEnc)[:, 0, :]

    print('==> Training the full network...')
    t_start = time.time()
    # Keep training until reach max iterations
    for epoch in range(max_iter):
        for i in range(0, num_training_vec, batch_size):
            end_ind = min(i + batch_size, num_training_vec)
            batch_x = X_train[i : end_ind]
            batch_y = y_train[i : end_ind]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((-1, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if (epoch + 1) % print_freq == 0:
            train_acc = accuracy.eval(feed_dict={x: X_train.reshape((-1, n_steps, n_input)),\
             y: y_train})
            train_acc_list.append(train_acc)
            val_acc = accuracy.eval(feed_dict={x: X_val.reshape((-1, n_steps, n_input)),\
             y: y_val})
            val_acc_list.append(val_acc)
            train_err = cost.eval(feed_dict={x: X_train.reshape((-1, n_steps, n_input)),\
             y: y_train})
            train_err_list.append(train_err)
            val_err = cost.eval(feed_dict={x: X_val.reshape((-1, n_steps, n_input)),\
             y: y_val})
            val_err_list.append(val_err)      
            print("-- epoch: %d, training error %g"%(epoch + 1, train_err))
            
    t_end = time.time()
    print('--Time elapsed for training: {t:.2f} \
        seconds'.format(t = t_end - t_start))

# ==============================================
#               RNN Evaluation
# ==============================================
# Reports
print('-- Training accuracy: {:.4f}'.format(train_acc_list[-1]))
print('-- Validation accuracy: {:.4f}'.format(val_acc_list[-1]))
print('-- Training error: {:.4E}'.format(train_err_list[-1]))
print('-- Validation error: {:.4E}'.format(val_err_list[-1]))

print('==> Generating error plot...')
x_list = range(0, print_freq * len(train_acc_list), print_freq)
train_err_plot = plt.plot(x_list, train_err_list, 'b-', label='training')
val_err_plot = plt.plot(x_list, val_err_list, '-', color='orange', label='validation')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs with {} Hidden Units'.format(n_hidden))
plt.legend(loc='best')
plt.savefig('exp6a_{}.png'.format(n_hidden), format='png')
plt.close()

print('==> Finished!')
